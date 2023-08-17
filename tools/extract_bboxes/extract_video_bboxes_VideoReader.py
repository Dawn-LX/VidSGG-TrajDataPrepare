# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 

import os
import argparse
import json
from tqdm import tqdm

import cv2
from decord import VideoReader
import torch
# import pickle
from PIL import Image

# from scene_graph_benchmark.scene_parser import SceneParser
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.data.datasets.utils.load_files import \
    config_dataset_file
from maskrcnn_benchmark.data.datasets.utils.load_files import load_labelmap_file
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from tools.demo.visual_utils import draw_bb


def NpyImg_to_Image(input_img):
    # the channel order of input_img is RGB
    img = Image.fromarray(input_img)
    return img


# refer to `tools/extract_bboxes/imread_vs_ViderReader.py`
class VideoDataset(object):
    def __init__(self,video_path,transforms):
        super().__init__()

        self.video_reader =  VideoReader(video_path)
        self.transforms = transforms


    def __len__(self):
        return len(self.video_reader)


    def __getitem__(self,idx):
        
        frame = self.video_reader[idx].asnumpy()  # this can not work when num_workers>0 (figure out why?)
        # frame = self.video_reader.get_batch([idx]).asnumpy().squeeze(0) # same problem as above
        # channel order: RGB
        
        _h = frame.shape[0]
        _w = frame.shape[1]
        wh = (_w,_h)
        img = NpyImg_to_Image(frame)
        img, _ = self.transforms(img, target=None)   # transforms 的最后有to_tensor 的操作

        return wh,img

def VidOR_videopath_to_videoname(video_path):
    tmp = video_path.split('/') # datasets/vidor-dataset/train_videos/0000/2401075277.mp4
    video_name = tmp[-2] + "_" + tmp[-1].split('.')[0]  # 0000_2401075277
    return video_name



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



def detect_objects_on_single_video(model, transforms, video_path, batch_size):
    # cv2_img is the original input, so we can get the height and 
    # width information to scale the output boxes.
    
    video_name = VidOR_videopath_to_videoname(video_path)

    dataset = VideoDataset(video_path,transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn = collator_func,
        num_workers = 0,
        drop_last=False,
        shuffle=False,
    )
    dataset_len = len(dataset)
    total_predictions = []
    for wh,batch_imgs in tqdm(dataloader,position=1, desc="{}".format(video_name), leave=False, ncols=160):
        with torch.no_grad():
            batch_predictions = model(batch_imgs)  
        wh = wh[0]
        batch_predictions = [p.to(torch.device("cpu")).resize(wh) for p in batch_predictions]  # 已经check过, resize 是 (width, height)
        total_predictions += batch_predictions ##################### 这部分代码能否优化一下，因为现在的gpu利用率不会长时间保持在较高百分比，是不是这里时间开销太多了？
        # if len(total_predictions) > 20:
        #     break
    
    assert len(total_predictions) == dataset_len

    total_results = []
    for prediction in total_predictions:
        boxes = prediction.bbox.tolist()  # len == num_bboxes
        classes = prediction.get_field("labels").tolist()
        scores = prediction.get_field("scores").tolist()

        total_results.append(
            {"boxes": boxes, "classes": classes, "scores": scores}
        )

    return total_results

def extract_signle_video_demo():

    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument("--config_file", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--labelmap_file", metavar="FILE",
                        help="labelmap file to select classes for visualizatioin")
    parser.add_argument("--batch_size", type=int,
                        help="batch size")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    video_path = "datasets/vidor-dataset/train_videos/0002/2693248308.mp4"
    video_name = VidOR_videopath_to_videoname(video_path)
   
    output_dir = "output/{}".format(video_name)
    mkdir(output_dir)

    
    assert cfg.MODEL.META_ARCHITECTURE == "AttrRCNN"
    model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)

    # dataset labelmap is used to convert the prediction to class labels
    dataset_labelmap_file = config_dataset_file(cfg.DATA_DIR,
                                                cfg.DATASETS.LABELMAP_FILE)
    assert dataset_labelmap_file
    dataset_allmap = json.load(open(dataset_labelmap_file, 'r'))
    dataset_labelmap = {int(val): key
                        for key, val in dataset_allmap['label_to_idx'].items()}
    

    # visual_labelmap is used to select classes for visualization
    try:
        # TODO: 先不考虑这个， 在应用Seq-NMS tracking的时候再考虑 visual_labelmap 过滤
        # 或者在 tracking 完成之后再用 visual_labelmap 过滤
        visual_labelmap = load_labelmap_file(args.labelmap_file)  
    except:
        visual_labelmap = None

    
    transforms = build_transforms(cfg, is_train=False)

    results = detect_objects_on_single_video(model,transforms,video_path,args.batch_size)

    results4save = {video_name:results}
    save_path = "output/{}_det_results.json".format(video_name)
    with open(save_path,'w') as f:
        json.dump(results4save,f)
    

    
    # visualize
    dataset = VideoDataset(video_path,transforms)
    print("saving results to: {}".format(output_dir))
    for fid, dets_per_img in enumerate(tqdm(results)):
        boxes = dets_per_img["boxes"]
        classes =  [dataset_labelmap[c_id] for c_id in  dets_per_img["classes"]]
        scores = dets_per_img["scores"]

        frame = dataset.video_reader[fid].asnumpy()  # 
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        draw_bb(frame, boxes, classes, scores)

        save_path = os.path.join(output_dir,"{:06d}_det.jpg".format(fid))
        
        cv2.imwrite(save_path, frame)
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument("--config_file", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--labelmap_file", metavar="FILE",
                        help="labelmap file to select classes for visualizatioin")
    parser.add_argument("--batch_size", type=int,
                        help="batch size")
    parser.add_argument("--start_id", type=int,help="batch size")
    parser.add_argument("--end_id_exclusive", type=int,help="batch size")
    parser.add_argument("--split", type=str,default="train",help="batch size")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    
    if args.split == "train":
        video_spit = "train_videos"
        output_dir = "output/VidOR_det_results/"
    elif args.split == "val":
        video_spit = "val_videos"
        output_dir = "output/VidORval_det_results/"
    else:
        raise ValueError

    video_dir =  "datasets/vidor-dataset/{}/".format(video_spit)
    
    group_ids = os.listdir(video_dir)
    video_paths_all = []
    for gid in group_ids:
        filenames = os.listdir(os.path.join(video_dir,gid))
        video_paths_all  += [os.path.join(video_dir,gid,filename)  for filename in filenames]
    video_paths_all = sorted(video_paths_all)
    
    if args.split == "train":
        assert len(video_paths_all) == 7000
    else:
        assert len(video_paths_all) == 835
    
    start_id = args.start_id
    end_id = args.end_id_exclusive
    video_paths = video_paths_all[start_id:end_id]  # datasets/vidor-dataset/train_videos/0000/2401075277.mp4

    #### filter out done videos
    video_paths_undo = []
    print("filter out done videos ...")
    for video_path in tqdm(video_paths):
        video_name = VidOR_videopath_to_videoname(video_path)
        save_path_ =  os.path.join(output_dir,"{}_det_results.json".format(video_name))
        if os.path.exists(save_path_):
            continue

        # if video_name == "1009_7416295940":  # this video CUDA out of memory
        #     continue
        
        video_paths_undo.append(video_path)
    print("{} videos left".format(len(video_paths_undo)))
    
    assert cfg.MODEL.META_ARCHITECTURE == "AttrRCNN"
    model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)

    # dataset labelmap is used to convert the prediction to class labels
    dataset_labelmap_file = config_dataset_file(cfg.DATA_DIR,
                                                cfg.DATASETS.LABELMAP_FILE)
    assert dataset_labelmap_file
    dataset_allmap = json.load(open(dataset_labelmap_file, 'r'))
    dataset_labelmap = {int(val): key
                        for key, val in dataset_allmap['label_to_idx'].items()}
    

    # visual_labelmap is used to select classes for visualization
    try:
        # TODO: 先不考虑这个， 在应用Seq-NMS tracking的时候再考虑 visual_labelmap 过滤
        # 或者在 tracking 完成之后再用 visual_labelmap 过滤
        visual_labelmap = load_labelmap_file(args.labelmap_file)  
    except:
        visual_labelmap = None

    
    transforms = build_transforms(cfg, is_train=False)
    desc_str = "{}[{}:{}]".format(video_spit,start_id,end_id)
    for video_path in tqdm(video_paths_undo, position=0, desc=desc_str, leave=True, ncols=160):
        video_name = VidOR_videopath_to_videoname(video_path)
        save_path =  os.path.join(output_dir,"{}_det_results.json".format(video_name))
        assert not os.path.exists(save_path)  # avoid over-write

        results = detect_objects_on_single_video(model,transforms,video_path,args.batch_size)

        results4save = {video_name:results}
        with open(save_path,'w') as f:
            json.dump(results4save,f)
    

    print("Done.")


def correct_json_results():
    json_dir = "output/VidOR_det_results"
    filenames = sorted(os.listdir(json_dir))
    print("num total json file: {}".format(len(filenames)))
    num_mismatch = 0
    for filename in tqdm(filenames):
        video_name_in_filename = filename.split("_det_")[0]
        path = os.path.join(json_dir,filename)
        with open(path,'r') as f:
            results = json.load(f)
        video_name = next(iter(results.keys()))
        if video_name_in_filename != video_name:
            print(filename,video_name)
            num_mismatch += 1
            src_path = os.path.join(json_dir,filename)
            dst_path = os.path.join(json_dir,"{}_det_results.json".format(video_name))
            os.system("mv {} {}".format(src_path,dst_path))
    
    print(num_mismatch)  # three mis-match files

def count_results_demo():
    video_exists = [0,2,3,4,5,6,8,9,10,11,12,14]
    video_exists = torch.as_tensor(video_exists)
    assert torch.all(torch.unique(video_exists) == video_exists)
    idx_diff = video_exists[1:] - video_exists[:-1]
    idx_jump = (idx_diff > 1).nonzero(as_tuple=True)[0]+1
    tmp = torch.as_tensor([0] + idx_jump.tolist() + [len(video_exists)])
    start_ids = tmp[:-1]
    range_lens = tmp[1:] - tmp[:-1]

    video_starts = video_exists[idx_jump]
    print(idx_diff)
    print(idx_jump)
    print(video_starts)
    print(range_lens)
    
    start_ids = start_ids.tolist()
    range_lens = range_lens.tolist()
    for s,len_ in zip(start_ids,range_lens):
        # print(s,len_,range_text,video_exists[s:s+len_])
        all_vids = video_exists[s:s+len_]
        s_vid = video_exists[s]
        e_vid = video_exists[s+len_-1]
        range_text = "videos[{}:{}],".format(s_vid,e_vid+1)
        if s_vid == e_vid:
            tmp = "i.e., [{}]".format(s_vid)
        else:
            tmp = "i.e., [{},...,{}]".format(s_vid,e_vid)
        print(range_text,tmp)
    
    # continuous_range = []
    # for s,len_ in zip(start_ids,range_lens):
    #     # print(s,len_,range_text,video_exists[s:s+len_])
    #     range_text = "videos[{}:{}],".format(video_exists[s],video_exists[s+len_-1]+1)
    #     print(range_text,"actual video ids:{}".format(video_exists[s:s+len_]))

def count_results():
    anno_dir = "/home/gkf/project/VidVRD_VidOR/vidor-dataset/annotation/training"
    group_ids = os.listdir(anno_dir)
    video_names_all = []
    for gid in group_ids:
        filenames = os.listdir(os.path.join(anno_dir,gid))
        video_names_all  += [gid + "_" + filename.split(".")[0]  for filename in filenames]

    video_names_all = sorted(video_names_all)
    
    results_dir = "output/VidOR_det_results"
    filenames = sorted(os.listdir(results_dir))
    
    # print(len(filenames))
    video_exists = []
    # 
    for idx,video_name in enumerate(video_names_all):
        result_path = os.path.join(results_dir,video_name+"_det_results.json")
        if os.path.exists(result_path):
            video_exists.append(idx)
    
    video_exists = torch.as_tensor(video_exists)
    print(video_exists)
    assert len(video_exists) == len(filenames)
    assert torch.all(torch.unique(video_exists) == video_exists)
    idx_diff = video_exists[1:] - video_exists[:-1]
    idx_jump = (idx_diff > 1).nonzero(as_tuple=True)[0]+1
    tmp = torch.as_tensor([0] + idx_jump.tolist() + [len(video_exists)])
    start_ids = tmp[:-1]
    range_lens = tmp[1:] - tmp[:-1]

    
    start_ids = start_ids.tolist()
    range_lens = range_lens.tolist()
    for s,len_ in zip(start_ids,range_lens):
        # print(s,len_,range_text,video_exists[s:s+len_])
        all_vids = video_exists[s:s+len_]
        s_vid = video_exists[s]
        e_vid = video_exists[s+len_-1]
        range_text = "videos[{}:{}],".format(s_vid,e_vid+1)
        if s_vid == e_vid:
            tmp = "i.e., [{}]".format(s_vid)
        else:
            tmp = "i.e., [{},...,{}]".format(s_vid,e_vid)
        print(range_text,tmp)
        
    print("total done videos: {}.".format(len(filenames)))


if __name__ == "__main__":
    
    # extract_signle_video_demo()
    main()
    # correct_json_results()

    '''
    python tools/extract_bboxes/extract_video_bboxes_VideoReader.py \
    --split train \
    --start_id 6944 \
    --end_id_exclusive 6950 \
    --config_file tools/extract_bboxes/vinvl_x152c4_extrat_bbox.yaml \
    --batch_size 2 \
    MODEL.DEVICE cuda:2 \
    MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth \
    MODEL.ROI_HEADS.NMS_FILTER 1 \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
    TEST.IGNORE_BOX_REGRESSION False


    ### train-set process summary:
    [0:500]    done
    [500:1000] 445 left, cuda-2 doing
    [1000:1500] cuda-3 doing
    [1500:2000] cuda-0 doing
    [2000:2500] cuda-1 doing


    ############ Val-set


    python tools/extract_bboxes/extract_video_bboxes_VideoReader.py \
    --split val \
    --start_id 0 \
    --end_id_exclusive 835 \
    --config_file tools/extract_bboxes/vinvl_x152c4_extrat_bbox.yaml \
    --batch_size 4 \
    MODEL.DEVICE cuda:0 \
    MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth \
    MODEL.ROI_HEADS.NMS_FILTER 1 \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
    TEST.IGNORE_BOX_REGRESSION False

    '''
