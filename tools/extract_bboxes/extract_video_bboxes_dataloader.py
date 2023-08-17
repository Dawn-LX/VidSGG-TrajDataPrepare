# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 

import os
import argparse
import json
from tqdm import tqdm

import cv2
import torch
import pickle
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


def cv2Img_to_Image(input_img):
    #### NOTE 注意： 这是个大坑： opencv 读进来的img不能直接用，要转成RGB；
    # 因为在detector训练的时候，用的是Image这个package读的图片，并转成RGB了，这一个操作是在Dataloader里面的：例如：
    # img = Image.open(os.path.join(self.root, path)).convert('RGB')
    ####

    # cv2_img = input_img.copy()
    cv2_img = input_img
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img

class Dataset_VidVRD(object):
    def __init__(self,imgs_dir,transforms):
        super().__init__()
        self.imgs_dir = imgs_dir
        self.transforms = transforms
        img_names = sorted(os.listdir(self.imgs_dir))
        self.img_paths = [os.path.join(self.imgs_dir,img_name) for img_name in img_names]

    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self,idx):
        cv2_img = cv2.imread(self.img_paths[idx])
        _h = cv2_img.shape[0]
        _w = cv2_img.shape[1]
        wh = (_w,_h)
        img = cv2Img_to_Image(cv2_img)
        img, _ = self.transforms(img, target=None)   # transforms 的最后有to_tensor 的操作

        return wh,img



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



def detect_objects_on_single_video(model, transforms, frames_dir, batch_size):
    # cv2_img is the original input, so we can get the height and 
    # width information to scale the output boxes.
    video_name = frames_dir.split('/')[-1]

    dataset = Dataset_VidVRD(frames_dir,transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn = collator_func,
        num_workers = 2,
        drop_last=False,
        shuffle=False,
    )
    dataset_len = len(dataset)
    dataloader_len = len(dataloader)
    # print("len(dataset)=={},batch_size=={},len(dataloader)=={},{}x{}={}".format(dataset_len,batch_size,dataloader_len,batch_size,dataloader_len,batch_size*dataloader_len))
    total_predictions = []
    for wh,batch_imgs in tqdm(dataloader,position=1, desc="{}".format(video_name), leave=False, ncols=160):
        with torch.no_grad():
            batch_predictions = model(batch_imgs)  
        wh = wh[0]
        batch_predictions = [p.to(torch.device("cpu")).resize(wh) for p in batch_predictions]  # 已经check过, resize 是 (width, height)
        total_predictions += batch_predictions
    
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

    video_name = "ILSVRC2015_train_00010018"
    frames_dir = "datasets/vidvrd-dataset/images/{}".format(video_name)
   
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

    results = detect_objects_on_single_video(model,transforms,frames_dir,args.batch_size)

    results4save = {video_name:results}
    save_path = "output/ILSVRC2015_train_00010018_det_results2.json"
    with open(save_path,'w') as f:
        json.dump(results4save,f)
    

    
    ## visualize
    # print("saving results to: {}".format(output_dir))
    # for fid, dets_per_img in enumerate(tqdm(results)):
    #     boxes = dets_per_img["boxes"]
    #     classes =  [dataset_labelmap[c_id] for c_id in  dets_per_img["classes"]]
    #     scores = dets_per_img["scores"]

    #     draw_bb(cv2_img_list[fid], boxes, classes, scores)

    #     save_path = os.path.join(output_dir,"{:06d}_det.jpg".format(fid))
        
    #     cv2.imwrite(save_path, cv2_img_list[fid])
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument("--config_file", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--labelmap_file", metavar="FILE",
                        help="labelmap file to select classes for visualizatioin")
    parser.add_argument("--batch_size", type=int,
                        help="batch size")
    parser.add_argument("--part_id", type=int,
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

    assert args.part_id in [1,2]

    video_dir =  "datasets/vidvrd-dataset/images"
    output_dir = "output/VidVRD_det_results"
    video_filenames = sorted(os.listdir(video_dir))
    frames_dir_all = [os.path.join(video_dir,filename) for filename in video_filenames]
    frames_dirs = []
    for frames_dir in frames_dir_all[500:]:
        video_name = frames_dir.split('/')[-1]
        save_path =  os.path.join("output/VidVRD_det_results_xsn","{}_det_results.json".format(video_name))  # 之前在肖博机器上跑过一些 （VidVRD_det_results_xsn这个文件夹现在已经删除了）
        if os.path.exists(save_path):
            continue
        frames_dirs.append(frames_dir)

    assert len(frames_dirs) == 500 - 407

    if args.part_id == 1:
        frames_dirs = frames_dirs[:45]
    else:
        frames_dirs = frames_dirs[45:]
    
    
    
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

    for frames_dir_per_video in tqdm(frames_dirs, position=0, desc="all videos", leave=False, ncols=160):
        video_name = frames_dir_per_video.split('/')[-1]
        save_path =  os.path.join(output_dir,"{}_det_results.json".format(video_name))
        if os.path.exists(save_path):
            print("{} exists, skip".format(save_path))
            continue
        
        results = detect_objects_on_single_video(model,transforms,frames_dir_per_video,args.batch_size)

        results4save = {video_name:results}
        with open(save_path,'w') as f:
            json.dump(results4save,f)
    

    print("Done.")

def invesgate_det_results():
    '''
    results fromat:
    det_results = {
        video_name:[
            res_1,
            res_2,
            ...
            res_T
        ]
    }, 
    where:
    video_name, e.g., "ILSVRC2015_train_00010018"
    T is the number of total frames
    res_i is like:
    res_i={
        "boxes":[[x1,y1,x2,y2],[x1,y1,x2,y2],...,[x1,y1,x2,y2]]
        "classes":[23,54,13,7,...,53]
        "scores": [0.12,0.45,0.98,...,0.87]
    }
    '''
    video_name = "ILSVRC2015_train_00010018"
    load_path = f"output/VidVRD_det_results/{video_name}_det_results.json"
    with open(load_path,'r') as f:
        det_results = json.load(f)
    
    det_results = det_results[video_name][0]
    boxes = det_results["boxes"]
    classes = det_results["classes"]
    scores = det_results["scores"]

    print(len(boxes),len(boxes[0]))

if __name__ == "__main__":
    
    # extract_signle_video_demo()
    # main()
    invesgate_det_results()
    '''
    python tools/extract_bboxes/extract_video_bboxes_dataloader.py \
    --config_file tools/extract_bboxes/vinvl_x152c4_extrat_bbox.yaml \
    --batch_size 4 \
    MODEL.DEVICE cuda:3 \
    MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth \
    MODEL.ROI_HEADS.NMS_FILTER 1 \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
    TEST.IGNORE_BOX_REGRESSION False

    python tools/extract_bboxes/extract_video_bboxes_dataloader.py \
    --config_file tools/extract_bboxes/vinvl_x152c4_extrat_bbox.yaml \
    --part_id 2 \
    --batch_size 4 \
    MODEL.DEVICE cuda:3 \
    MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth \
    MODEL.ROI_HEADS.NMS_FILTER 1 \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
    TEST.IGNORE_BOX_REGRESSION False

    
    '''
