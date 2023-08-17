# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 

import cv2
import torch
import numpy as np
from tqdm.std import tqdm
from torch_scatter import scatter
import os
import argparse
import json
from collections import defaultdict
from PIL import Image

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


def cv2Img_to_Image(input_img):
    cv2_img = input_img.copy()
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img

class DatasetVidVRDTraj(object):
    def __init__(self,imgs_dir,video_name,transforms,frame2boxes,frame2tids):
        super().__init__()
        
        self.imgs_dir = imgs_dir
        self.video_name = video_name
        self.transforms = transforms
        self.frame2boxes = frame2boxes
        self.frame2tids = frame2tids
        self.frame_ids = list(frame2boxes.keys())
        
       
    def __len__(self):
        return len(self.frame_ids)


    def __getitem__(self,idx):
        frame_id = self.frame_ids[idx]
        img_path = os.path.join(self.imgs_dir,self.video_name,"{:06d}.JPEG".format(frame_id)) 
        tids_list = self.frame2tids[frame_id]
        bboxes_list = self.frame2boxes[frame_id]

        cv2_img = cv2.imread(img_path)

        ori_height = cv2_img.shape[0]
        ori_width = cv2_img.shape[1]
        ori_wh = (ori_width,ori_height)
        img_input = cv2Img_to_Image(cv2_img)
        img_input, _ = self.transforms(img_input, target=None)
        input_h,input_w = img_input.shape[1],img_input.shape[2]
        input_wh = (input_w,input_h)

        bboxes = BoxList(bboxes_list,ori_wh,mode='xyxy')
        bboxes = bboxes.resize(input_wh)

        return img_input,bboxes,tids_list


def collator_func(batch):

    """
    batch is a list ,len(batch) == batch_size
    batch[i] is a tuple, batch[i][0],batch[i][1] is wh, img, respectively
    This function should be passed to the torch.utils.data.DataLoader
    """
    # batch_size = len(batch)
    wh = [b[0] for b in batch]
    batch_imgs = [b[1] for b in batch]

    return wh,batch_imgs

def extract_feature_given_bbox(model, transforms, cv2_img, bboxes):
    
    # cv2_img is the original input, so we can get the height and 
    # width information to scale the output boxes.
    ori_height = cv2_img.shape[0]
    ori_width = cv2_img.shape[1]
    ori_wh = (ori_width,ori_height)
    img_input = cv2Img_to_Image(cv2_img)
    img_input, _ = transforms(img_input, target=None)
    input_h,input_w = img_input.shape[1],img_input.shape[2]
    input_wh = (input_w,input_h)

    bboxes = BoxList(bboxes,ori_wh,mode='xyxy')
    bboxes = bboxes.resize(input_wh)

    ''' in AttrRCNN.forward
    images = to_image_list(images)
    images = images.to(self.device)
    features = self.backbone(images.tensors)

    proposals, proposal_losses = self.rpn(images, features, targets)
    x, predictions, detector_losses = self.roi_heads(features,proposals, targets)

    '''
    # avg_pooler = torch.nn.AdaptiveAvgPool2d(1)

    with torch.no_grad():
        images = to_image_list(img_input)
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
        # bbox_features = avg_pooler(bbox_features)    # (num_boxes, 2048, 1, 1)
        # bbox_features = bbox_features.reshape(bbox_features.size(0),-1)
    
    print(bbox_features.shape,"bbox_features.shape")
    # assert False
    bbox_features = bbox_features.to(torch.device("cpu"))

    
    return bbox_features


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


def extract_feature_per_seg(model, transforms, frame2boxes, frame2tids,video_name,imgs_dir):
    dataset = DatasetVidVRDTraj(imgs_dir,video_name, transforms, frame2boxes, frame2tids)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn = lambda x : x ,
        num_workers = 2,
        drop_last=False,
        shuffle=False,
    )

    total_box_features = []
    total_tid_list = []
    for batch_data in tqdm(dataloader,position=1, desc="{}".format(video_name), leave=False, ncols=160):
        img_input,bboxes,tids_list = batch_data[0]
        bbox_features = extract_feature_given_bbox_v2(model, img_input, bboxes)
        # bbox_features.shape == (num_boxes,2048)  num_boxes == len(tids_list)

        total_box_features.append(bbox_features)
        total_tid_list += tids_list

    total_tids = torch.tensor(total_tid_list,dtype=torch.long)  # shape == (N_boxes,)
    total_tids = total_tids[:,None]  # shape == (N_boxes, 1)
    total_box_features = torch.cat(total_box_features,dim=0) # (N_boxes, 2048)    
    bbox_features = scatter(total_box_features,total_tids,dim=0,reduce='mean')  # shape == (num_trajs, 2048)
    
    return bbox_features



def draft():
    imgs_dir = "datasets/vidvrd-dataset/images"
    filename = "ILSVRC2015_train_00010001-0016-0048.json"
    input_path = "/home/gkf/project/VidVRD-II/tracklets_results/VidVRD_segment_tracking_results/{}".format(filename)
    '''
    tracking_results = [
        {
            'fstart': int(fstart),      # relative frame idx w.r.t this segment
            'score': score,             # float scalar
            'bboxes': bboxes.tolist()   # list[list], len == num_frames
        },
        {
            'fstart': int(fstart),      # relative frame idx w.r.t this segment
            'score': score,             # float scalar
            'bboxes': bboxes.tolist()   # list[list], len == num_frames
        },
        ...
    ]  # len(tracking_results) == num_tracklets
    '''
    with open(input_path,'r') as f:
        tracking_results = json.load(f)
    
    video_name, fstrat, fend = filename.split('-')
    fstrat,fend = int(fstrat),int(fend.split('.')[0])

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
    
    tid2features = defaultdict(list)
    total_box_features = []
    total_tid_list = []
    for frame_id, bboxes_list in frame2boxes.items():  
        tids_list = frame2tids[frame_id]
        img_path = os.path.join(imgs_dir,video_name,"{:06d}.JPEG".format(frame_id))

        cv2_img = cv2.imread(img_path)
        bbox_features = extract_feature_given_bbox(model, transforms, cv2_img, bboxes_list)
        # bbox_features: list[list], shape == (num_boxes,2048)  num_boxes == len(tids_list)
        for idx,tid in enumerate(tids_list):
            tid2features[tid].append(bbox_features[idx])
        
        total_box_features.append(bbox_features)
        total_tid_list += tids_list

    for tid, feature_list in tid2features.items():
        feature = torch.tensor(feature_list).mean(dim=0)  # shape == (num_boxes,2048) --> (2048,)
    
    # TODO avoid for loop, use edge_scatter
    total_box_features = torch.cat(total_box_features,dim=0) # (N_boxes, 2048)
    bbox_features = edge_scatter(total_tid_list,total_box_features,'mean')

    


    # TODO use dataloader， 在 __init__ 中先构建好 img_paths
    # frame_id_list = list(frame2boxes.keys())
    # img_paths = [os.path.join(imgs_dir,video_name,"{:06d}.JPEG".format(fid)) for fid in frame_id_list]
    # 然后 __getiitem__ 应该返回 bboxes_list 和 cv2_img （在 __getiitem__ 中处理好 BoxList的构建 和 resize
    # 对每个segement 构建一个dataloader， 还是对每个video 构建一个dataloader ？ 感觉每个segment一个dataloader比较好

    # 能不能只构建一个整个的 dataloader  ？ 好像不太行， 因为我们存的时候应该是每个 segment存一个json 文件


def single_segment_demo():
    
    
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument("--config_file", metavar="FILE",
                        help="path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)

    assert cfg.MODEL.META_ARCHITECTURE == "AttrRCNN"
    model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)
    transforms = build_transforms(cfg, is_train=False)


    imgs_dir = "datasets/vidvrd-dataset/images"
    filename = "ILSVRC2015_train_00010001-0016-0048.json"
    input_path = "/home/gkf/project/VidVRD-II/tracklets_results/VidVRD_segment_tracking_results/{}".format(filename)
    '''
    tracking_results = [
        {
            'fstart': int(fstart),      # relative frame idx w.r.t this segment
            'score': score,             # float scalar
            'bboxes': bboxes.tolist()   # list[list], len == num_frames
        },
        {
            'fstart': int(fstart),      # relative frame idx w.r.t this segment
            'score': score,             # float scalar
            'bboxes': bboxes.tolist()   # list[list], len == num_frames
        },
        ...
    ]  # len(tracking_results) == num_tracklets
    '''
    with open(input_path,'r') as f:
        tracking_results = json.load(f)
    
    video_name, fstrat, fend = filename.split('-')
    fstrat,fend = int(fstrat),int(fend.split('.')[0])

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

    total_box_features = []
    total_tid_list = []
    for frame_id, bboxes_list in frame2boxes.items():  
        tids_list = frame2tids[frame_id]
        img_path = os.path.join(imgs_dir,video_name,"{:06d}.JPEG".format(frame_id))

        cv2_img = cv2.imread(img_path)
        bbox_features = extract_feature_given_bbox(model, transforms, cv2_img, bboxes_list)
        # bbox_features.shape == (num_boxes,2048)  num_boxes == len(tids_list)

        total_box_features.append(bbox_features)
        total_tid_list += tids_list

    total_tids = torch.tensor(total_tid_list,dtype=torch.long)  # shape == (N_boxes,)
    total_tids = total_tids[:,None]  # shape == (N_boxes, 1)
    total_box_features = torch.cat(total_box_features,dim=0) # (N_boxes, 2048)
    bbox_features = scatter(total_box_features,total_tids,dim=0,reduce='mean')  # shape == (num_trajs, 2048)
    assert bbox_features.shape == (num_trajs, 2048)
    
    # TODO use dataloader， 在 __init__ 中先构建好 img_paths
    # frame_id_list = list(frame2boxes.keys())
    # img_paths = [os.path.join(imgs_dir,video_name,"{:06d}.JPEG".format(fid)) for fid in frame_id_list]
    # 然后 __getiitem__ 应该返回 bboxes_list 和 cv2_img （在 __getiitem__ 中处理好 BoxList的构建 和 resize
    # 对每个segement 构建一个dataloader， 还是对每个video 构建一个dataloader ？ 感觉每个segment一个dataloader比较好

    # 能不能只构建一个整个的 dataloader  ？ 好像不太行， 因为我们存的时候应该是每个 segment存一个json 文件

    print(bbox_features.shape)
    print(bbox_features[:2,:10])
    # bbox_features = bbox_features.cpu().numpy()
    # np.save("output/X152C5_test/bbox_features.npy",bbox_features)

    
    
    print("finish")


def main():
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument("--config_file", metavar="FILE",
                        help="path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)

    assert cfg.MODEL.META_ARCHITECTURE == "AttrRCNN"
    model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)
    transforms = build_transforms(cfg, is_train=False)


    imgs_dir = "datasets/vidvrd-dataset/images"
    tracking_res_dir = "/home/gkf/project/VidVRD-II/tracklets_results/VidVRD_segment_tracking_results/"
    tracking_res_names = sorted(os.listdir(tracking_res_dir))

    for filename in tqdm(tracking_res_names, position=0, desc="all segments", leave=False, ncols=160):
        input_path = os.path.join(tracking_res_dir,filename) 
        with open(input_path,'r') as f:
            tracking_results = json.load(f)
        '''
            tracking_results = [
                {
                    'fstart': int(fstart),      # relative frame idx w.r.t this segment
                    'score': score,             # float scalar
                    'bboxes': bboxes.tolist()   # list[list], len == num_frames
                },
                {
                    'fstart': int(fstart),      # relative frame idx w.r.t this segment
                    'score': score,             # float scalar
                    'bboxes': bboxes.tolist()   # list[list], len == num_frames
                },
                ...
            ]  # len(tracking_results) == num_tracklets
        '''
        video_name, fstrat, fend = filename.split('-') # e.g., "ILSVRC2015_train_00010001-0016-0048.json"
        fstrat,fend = int(fstrat),int(fend.split('.')[0])
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
            model, transforms, frame2boxes, frame2tids,video_name,imgs_dir
        )
        assert bbox_features.shape == (num_trajs, 2048)
    
        bbox_features = bbox_features.cpu().numpy()
        np.save("output/X152C5_test/bbox_features.npy",bbox_features)

    
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

    


if __name__ == "__main__":
   
    single_segment_demo()
    # scatter_demo()
    # 这两个输出的 RoI feature不一样，这是因为对feature_extractor输入的bbox不一样
    # 在ImgOnly_demo()中这个bbox是 rpn的proposal， 
    # 但是在ImgGivenBox_demo()中， 这个box是最终预测的box，与rpn给出的box有查别，但是比rpn的box更准一点，
    # 所以ImgGivenBox_demo()输出的roi feature 没有问题，甚至更好

    '''
    python tools/extract_features/extract_traj_features_demo.py \
    --config_file sgg_configs/vgattr/vinvl_x152c4.yaml \
    MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth \
    MODEL.ROI_HEADS.NMS_FILTER 1 \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
    TEST.IGNORE_BOX_REGRESSION False \
    TEST.OUTPUT_FEATURE True
    '''
