# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch_scatter import scatter
import os
import argparse
import json
from collections import defaultdict
from PIL import Image
from decord import VideoReader

from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.data.datasets.utils.load_files import \
    config_dataset_file
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def NpyImg_to_Image(input_img):
    # the channel order of input_img is RGB
    img = Image.fromarray(input_img)
    return img

def VidOR_videopath_to_videoname(video_path):
    tmp = video_path.split('/') # datasets/vidor-dataset/train_videos/0000/2401075277.mp4
    video_name = tmp[-2] + "_" + tmp[-1].split('.')[0]  # 0000_2401075277
    return video_name

class DatasetTraj_VideoReader(object):
    def __init__(self,video_path,transforms,seg_paths):
        super().__init__()
        
        self.video_reader =  VideoReader(video_path)
        self.transforms = transforms
        self.seg_paths = seg_paths
        self.video_name = VidOR_videopath_to_videoname(video_path)

        tid_wrt_video = 0
        num_trajs = []
        frame2boxes = defaultdict(list)
        frame2tids = defaultdict(list)
        for seg_path in self.seg_paths:
            # e.g., seg_path = .../VidORtrain_tracking_results_gt/0000_2401075277/0000_2401075277-0000-0030.json
            seg_tag = seg_path.split('/')[-1].split('.')[0]  # e.g., 0000_2401075277-0000-0030
            video_name, fstrat, fend = seg_tag.split('-')
            assert self.video_name == video_name, "{},{}".format(self.video_name,video_name)
            fstrat,fend = int(fstrat),int(fend)

            with open(seg_path,'r') as f:
                tracking_results = json.load(f)        
        
            for tracklet in tracking_results:  # tracking_results can be empty
                relative_fstrat = tracklet["fstart"] # w.r.t the start fid of this segment
                score = tracklet["score"]
                bboxes = tracklet["bboxes"]
                for idx,bbox in enumerate(bboxes):
                    frame_id = fstrat + relative_fstrat + idx
                    frame2boxes[frame_id].append(bbox)
                    frame2tids[frame_id].append(tid_wrt_video)
                tid_wrt_video += 1
        
            num_trajs.append(len(tracking_results))
        assert tid_wrt_video == sum(num_trajs)

        self.frame_ids = list(frame2boxes.keys())
        self.frame2boxes = frame2boxes
        self.frame2tids = frame2tids
        self.num_trajs = num_trajs



    def __len__(self):
        return len(self.frame_ids)


    def __getitem__(self,idx):
        frame_id = self.frame_ids[idx]
        tids_list = self.frame2tids[frame_id]
        bboxes_list = self.frame2boxes[frame_id]

        
        frame = self.video_reader[frame_id].asnumpy()  # this can not work when num_workers>0 (figure out why?)
        # frame = self.video_reader.get_batch([idx]).asnumpy().squeeze(0) # same problem as above
        # channel order: RGB
        
        ori_height = frame.shape[0]
        ori_width = frame.shape[1]
        ori_wh = (ori_width,ori_height)
        img_input = NpyImg_to_Image(frame)
        img_input, _ = self.transforms(img_input, target=None)
        input_h,input_w = img_input.shape[1],img_input.shape[2]
        input_wh = (input_w,input_h)

        bboxes = BoxList(bboxes_list,ori_wh,mode='xyxy')
        bboxes = bboxes.resize(input_wh)

        return img_input,bboxes,tids_list




def extract_feature_given_bbox_v2(model, transformed_img, bboxes):
    

    ''' in AttrRCNN.forward
    images = to_image_list(images)
    images = images.to(self.device)
    features = self.backbone(images.tensors)

    proposals, proposal_losses = self.rpn(images, features, targets)
    x, predictions, detector_losses = self.roi_heads(features,proposals, targets)

    '''

    with torch.no_grad():
        images = to_image_list(transformed_img)
        images = images.to(model.device)
        bboxes = bboxes.to(model.device)
        features = model.backbone(images.tensors) 
        # features: list[tensor], len == 1 for ResNet-C4 backbone, for FPN, len == num_levels
        # features[0].shape == (batch_size, 1024, H/16, W/16), (W,H) == input_wh

        ''' original code in forward function of model
        proposals, proposal_losses = model.rpn(images, features, targets)
        x = model.roi_heads.box.feature_extractor(features, proposals)

        # proposals is a list of `BoxList` objects (mode=='xyxy'), with filed 'objectness', objectness 在 train RPN的时候用到，现在inference的时候用不到
            # len(proposals) == batch_size (i.e, number of imgs)
            # proposals[0].bbox  is w.r.t the resized image input (NOTE not normalized to 0~1), where the resize resolution is determined by `transforms`
            # proposals[0].bbox.shape == (300,4), where 300 is determined by MODEL.RPN.POST_NMS_TOP_N_TEST
        # proposal_losses is an empty dict() because we are in test mode
        # x.shape == (300, 2048,7,7), 300 is the number of bboxes

        the type of feature_extractor is controlled by cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR, default: ResNet50Conv5ROIFeatureExtractor
        '''
        # print(len(features),features[0].shape)
        proposals = [bboxes]  # because num_imgs == 1
        bbox_features = model.roi_heads.box.feature_extractor(features, proposals)  # 
        # print(bbox_features.shape)  # (num_boxes, 2048, 7, 7); 
        # NOTE num_boxes is the total number of bboxes in this batch of images, where the order is determined by the order of list (proposal)

        bbox_features = bbox_features.mean([2,3])  # (num_boxes, 2048)
    
    # assert False
    bbox_features = bbox_features.to(torch.device("cpu"))

    
    return bbox_features


def extract_feature_per_video(model, transforms, video_path, seg_paths):
    dataset = DatasetTraj_VideoReader(video_path,transforms,seg_paths)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn = lambda x : x[0] ,
        num_workers = 0,
        drop_last=False,
        shuffle=False,
    )

    total_box_features = []
    total_tid_list = []
    for img_input,bboxes,tids_list in tqdm(dataloader,position=1, desc="{}".format(dataset.video_name), leave=False, ncols=160):
        
        bbox_features = extract_feature_given_bbox_v2(model, img_input, bboxes)
        # bbox_features.shape == (num_boxes,2048)  num_boxes == len(tids_list)

        total_box_features.append(bbox_features)
        total_tid_list += tids_list

    total_tids = torch.tensor(total_tid_list,dtype=torch.long)  # shape == (N_boxes,)
    total_tids = total_tids[:,None]  # shape == (N_boxes, 1), tid w.r.t video
    total_box_features = torch.cat(total_box_features,dim=0) # (N_boxes, 2048)    
    traj_features = scatter(total_box_features,total_tids,dim=0,reduce='mean')  # shape == (N_trajs, 2048)
    # N_traj is the total trajs in this video
    assert traj_features.shape[0] == sum(dataset.num_trajs)
    
    return traj_features



def main():
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument("--config_file", metavar="FILE",help="path to config file")
    parser.add_argument("--input_dir", type=str,help="input_dir")
    parser.add_argument("--split", type=str,default="train",help="batch size")
    parser.add_argument("--start_id", type=int,help="batch size")
    parser.add_argument("--end_id_exclusive", type=int,help="batch size")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    assert args.split in ("train","val")

    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)

    tracking_res_dir = args.input_dir  # e.g., "/home/gkf/project/VidVRD-II/tracklets_results/VidORtrain_tracking_results_gt"
    video_dir =  "datasets/vidor-dataset/{}_videos/".format(args.split)

    group_ids = os.listdir(video_dir)
    video_names_all = []
    for gid in group_ids:
        filenames = os.listdir(os.path.join(video_dir,gid))
        video_names_all  += [gid + "_" + filename.split(".")[0]  for filename in filenames]

    video_names_all = sorted(video_names_all)

    if args.split == "train":
        assert len(video_names_all) == 7000
    else:
        assert len(video_names_all) == 835
    # print(video_names_all[:10])
    start_id = args.start_id
    end_id = args.end_id_exclusive
    video_names = video_names_all[start_id:end_id] 

    tracking_res_paths_all = []
    videoname2paths = dict()
    for video_name in video_names:
        filenames = sorted(os.listdir(os.path.join(tracking_res_dir,video_name)))
        videoname2paths[video_name] =  [os.path.join(tracking_res_dir,video_name,filename)  for filename in filenames]
    tracking_res_paths_all = sorted(tracking_res_paths_all)  # inlcude all segment
    # e.g., .../VidORtrain_tracking_results_gt/0000_2401075277/0000_2401075277-0000-0030.json


    #### filter out done videos
    input_paths_undo = dict()
    print("filter out done videos ...")
    for video_name,paths in videoname2paths.items():
        save_path_ =  os.path.join(output_dir,"{}_traj_features.npy".format(video_name))
        if os.path.exists(save_path_):
            continue
        input_paths_undo[video_name] = paths

    print("{} videos left".format(len(input_paths_undo)))


    assert cfg.MODEL.META_ARCHITECTURE == "AttrRCNN"
    model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)
    transforms = build_transforms(cfg, is_train=False)

    desc_str = "{}_videos[{}:{}]".format(args.split,start_id,end_id)
    for video_name,seg_paths in tqdm(input_paths_undo.items(), position=0, desc=desc_str, leave=True, ncols=160):
        gid,vid = video_name.split('_')
        video_path = os.path.join(video_dir,gid,vid+".mp4")
        save_path =  os.path.join(output_dir,"{}_traj_features.npy".format(video_name))

        traj_features = extract_feature_per_video(model,transforms,video_path,seg_paths)
        # traj_features = torch.split(traj_features,split_list,dim=0)  # Not split, we save the all seg-trajs in the video as one file
    
        traj_features = traj_features.cpu().numpy()
        np.save(save_path,traj_features)

    
    print("finish")


def extract_feature_per_seg(model, transforms, frame2boxes, frame2tids,video_reader):
    
    frame_ids = sorted(list(frame2boxes.keys()))

    total_box_features = []
    total_tid_list = []
    for fid in tqdm(frame_ids,position=1, desc="segments", leave=False, ncols=160):
        bboxes_list = frame2boxes[fid]
        tids_list = frame2tids[fid]

        ##########
        frame = video_reader[fid].asnumpy()  # channel order: RGB
        
        ori_height = frame.shape[0]
        ori_width = frame.shape[1]
        ori_wh = (ori_width,ori_height)
        img_input = NpyImg_to_Image(frame)
        img_input, _ = transforms(img_input, target=None)
        input_h,input_w = img_input.shape[1],img_input.shape[2]
        input_wh = (input_w,input_h)

        bboxes = BoxList(bboxes_list,ori_wh,mode='xyxy')
        bboxes = bboxes.resize(input_wh)
        ########## 


        bbox_features = extract_feature_given_bbox_v2(model, img_input, bboxes)
        # bbox_features.shape == (num_boxes,2048)  num_boxes == len(tids_list)

        total_box_features.append(bbox_features)
        total_tid_list += tids_list

    total_tids = torch.tensor(total_tid_list,dtype=torch.long)  # shape == (N_boxes,)
    total_tids = total_tids[:,None]  # shape == (N_boxes, 1)
    total_box_features = torch.cat(total_box_features,dim=0) # (N_boxes, 2048)    
    bbox_features = scatter(total_box_features,total_tids,dim=0,reduce='mean')  # shape == (num_trajs, 2048)
    
    return bbox_features


def single_video_demo():
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument("--config_file", metavar="FILE",help="path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = "output/0000_2401075277_gtTrajFeats/"
    mkdir(output_dir)

    video_name = "0000_2401075277"
    tracking_res_dir = "/home/gkf/project/VidVRD-II/tracklets_results/VidORtrain_tracking_results_gt/0000_2401075277"
    video_path = "datasets/vidor-dataset/train_videos/0000/2401075277.mp4"
    video_reader =  VideoReader(video_path)
  
    filenames = sorted(os.listdir(tracking_res_dir))
    tracking_res_paths =  [os.path.join(tracking_res_dir,filename)  for filename in filenames]
    tracking_res_paths = sorted(tracking_res_paths)  # inlcude all segment
    # e.g., .../VidORtrain_tracking_results_gt/0000_2401075277/0000_2401075277-0000-0030.json

    


    assert cfg.MODEL.META_ARCHITECTURE == "AttrRCNN"
    model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)
    transforms = build_transforms(cfg, is_train=False)

    
    for seg_path in tqdm(tracking_res_paths,position=0,desc=video_name):
        seg_tag = seg_path.split('/')[-1].split('.')[0]
        save_path =  os.path.join(output_dir,"{}.npy".format(seg_tag))


        video_name_, fstrat, fend = seg_tag.split('-')
        assert video_name == video_name_
        fstrat,fend = int(fstrat),int(fend)

        with open(seg_path,'r') as f:
            tracking_results = json.load(f)        
        
        frame2boxes = defaultdict(list)
        frame2tids = defaultdict(list)
        for tid,tracklet in enumerate(tracking_results):
            relative_fstrat = tracklet["fstart"] # w.r.t the start fid of this segment
            score = tracklet["score"]
            bboxes = tracklet["bboxes"]
            for idx,bbox in enumerate(bboxes):
                frame_id = fstrat + relative_fstrat + idx
                frame2boxes[frame_id].append(bbox)
                frame2tids[frame_id].append(tid)
    
        num_trajs = len(tracking_results)
        bbox_features = extract_feature_per_seg(
            model, transforms, frame2boxes, frame2tids,video_reader
        )
        assert bbox_features.shape == (num_trajs, 2048)
    
        bbox_features = bbox_features.cpu().numpy()
        np.save(save_path,bbox_features)

    
    print("finish")



def sample_demo():
    # NOTE 不用sample了，直接一整个轨迹都输进去。然后对32帧的 bbox_feature 取平均
    num_sample_box = 5
    L = 9
    # boxes = torch.randint(9,99,size=(L,)).tolist()
    boxes = list(range(L))
    print(boxes)
    interval = L // num_sample_box
    print(interval,boxes[0:L:interval])

def scatter_demo():
    # from torch_scatter import scatter
    device = torch.device("cuda:0")
    dim_msg = 9
    msg_recv = torch.randint(1,9,size=(5,dim_msg)).float().to(device)
    index = torch.tensor([1,2,3,2,1]).to(device)

    index = index[:,None] # broadcasting
    print(index)
    print(msg_recv)
    msg_recv1 = scatter(msg_recv,index,dim=0,reduce="mean")
    print(msg_recv1)

    index2 = index.repeat(1,dim_msg)
    msg_recv2 = scatter(msg_recv,index2,dim=0,reduce="mean")
    print(msg_recv2)


def count_demo():
    xx = "output/VidVRD_tracking_results_gt"
    xx = os.listdir(xx)
    print(len(xx))

def SegNpy_vs_VideoNpy():
    
    traj_feat_path2 = "output/VidORtrain_gt_traj_features/0000_2401075277_traj_features.npy"
    total_features = np.load(traj_feat_path2)
    print(total_features.shape)

    traj_feat_dir1 = "output/0000_2401075277_gtTrajFeats"
    filenames = os.listdir(traj_feat_dir1)
    seg_feat_paths = [os.path.join(traj_feat_dir1,name) for name in sorted(filenames)]

    tracking_res_dir = "/home/gkf/project/VidVRD-II/tracklets_results/VidORtrain_tracking_results_gt/0000_2401075277"
    filenames = sorted(os.listdir(tracking_res_dir))
    seg_track_paths =  [os.path.join(tracking_res_dir,filename)  for filename in filenames]
    assert len(seg_feat_paths) == len(seg_track_paths)
    global_tid = 0
    for feat_path,track_path in zip(seg_feat_paths,seg_track_paths):
        seg_traj_features = np.load(feat_path)
        with open(track_path,'r') as f:
            seg_track_res = json.load(f)
        num_trajs = len(seg_track_res)
        assert seg_traj_features.shape == (num_trajs,2048)

        features = total_features[global_tid:global_tid+num_trajs,:]
        global_tid += num_trajs

        print(features[:,0])
        print(seg_traj_features[:,0])
        eq_sum = (features == seg_traj_features).sum()
        print(eq_sum,num_trajs*2048)
    print(global_tid)

def lenght_demo():
    xx = "/home/gkf/project/VidVRD-II/tracklets_results/VidORval_tracking_results"
    xx = sorted(os.listdir(xx))
    len_ = len(xx)

    # input_dir = "/home/gkf/project/scene_graph_benchmark/output/VidORval_det_results"
    # input_names = sorted(os.listdir(input_dir))
    # for name in input_names:
    #     x = name.split("_det_")[0]
    #     print(x)
    count = 0
    for x in xx:
        print(x)
        gid = x.split("_")[0]
        if int(gid)>=1016:
            count+=1
    print(len_,count,len_-count)


def filter_traj_features():
    '''
    refer to `/home/gkf/project/VidVRD-II/video_object_detection/object_bbox2traj_segment.py`, func:`filter_track_res_and_feature`
    '''

if __name__ == "__main__":

    # lenght_demo()
    main()
    # SegNpy_vs_VideoNpy()
    '''
    python tools/extract_features/extract_traj_features_VideoReader.py \
    --config_file sgg_configs/vgattr/vinvl_x152c4.yaml \
    MODEL.DEVICE cuda:3 \
    MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth \
    MODEL.ROI_HEADS.NMS_FILTER 1 \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
    TEST.IGNORE_BOX_REGRESSION False \
    TEST.OUTPUT_FEATURE True
    '''



    '''for extract traj features of VidOR-train gt_trajs:

    NOTE : TODO: sort by video length
    
    python tools/extract_features/extract_traj_features_VideoReader.py \
    --config_file sgg_configs/vgattr/vinvl_x152c4.yaml \
    --input_dir /home/gkf/project/VidVRD-II/tracklets_results/VidORtrain_tracking_results_th-15-5 \
    --split train \
    --start_id 6750 \
    --end_id_exclusive 7000 \
    MODEL.DEVICE cuda:1 \
    MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth \
    MODEL.ROI_HEADS.NMS_FILTER 1 \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
    TEST.IGNORE_BOX_REGRESSION False \
    TEST.OUTPUT_FEATURE True \
    OUTPUT_DIR output/VidORtrain_traj_features_th-15-5

    ### for extract vidor-val 
    [0:450] done
    [450:550] cuda-1
    [550:650] cuda-2
    [650:750] cuda-3
    [750:835] cuda-0

    python tools/extract_features/extract_traj_features_VideoReader.py \
    --config_file sgg_configs/vgattr/vinvl_x152c4.yaml \
    --input_dir /home/gkf/project/VidVRD-II/tracklets_results/VidORval_tracking_results \
    --split val \
    --start_id 680 \
    --end_id_exclusive 835 \
    MODEL.DEVICE cuda:3 \
    MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth \
    MODEL.ROI_HEADS.NMS_FILTER 1 \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
    TEST.IGNORE_BOX_REGRESSION False \
    TEST.OUTPUT_FEATURE True \
    OUTPUT_DIR output/tmp


    ### for extract vidor-val gt

    python tools/extract_features/extract_traj_features_VideoReader.py \
    --config_file sgg_configs/vgattr/vinvl_x152c4.yaml \
    --input_dir /home/gkf/project/VidVRD-II/tracklets_results/VidORval_tracking_results_gt \
    --split val \
    --start_id 600 \
    --end_id_exclusive 835 \
    MODEL.DEVICE cuda:2 \
    MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth \
    MODEL.ROI_HEADS.NMS_FILTER 1 \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
    TEST.IGNORE_BOX_REGRESSION False \
    TEST.OUTPUT_FEATURE True \
    OUTPUT_DIR output/VidORval_gt_traj_features

    '''
