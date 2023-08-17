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

def VidRead2ImgNpyLits(video_path):
    #### NOTE 注意： 这是个大坑： opencv 是BGR的， 读进来的img不能直接用，要转成RGB；
    # 因为在detector训练的时候，用的是Image这个package读的图片，并转成RGB了，这一个操作是在Dataloader里面的：例如：
    # img = Image.open(os.path.join(self.root, path)).convert('RGB')
    ####

    img_list = []
    cap = cv2.VideoCapture(video_path)  ##打开视频文件
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success = True
    count = 0
    while success and count < n_frames:
        success, image = cap.read()
        if success:
            # image.shape == (H,W,3)
            img_list.append(image)  
            count+=1
    assert count == n_frames
    return img_list


def cv2Img_to_Image(input_img):
    #### NOTE 注意： 这是个大坑： opencv 读进来的img不能直接用，要转成RGB；
    # 因为在detector训练的时候，用的是Image这个package读的图片，并转成RGB了，这一个操作是在Dataloader里面的：例如：
    # img = Image.open(os.path.join(self.root, path)).convert('RGB')
    ####

    cv2_img = input_img.copy()
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img


def detect_objects_on_single_video(model, transforms, cv2_img_list, batch_size):
    # cv2_img is the original input, so we can get the height and 
    # width information to scale the output boxes.
    cv2_img = cv2_img_list[0]
    _h = cv2_img.shape[0]
    _w = cv2_img.shape[1]
    _wh = (_w,_h)
    num_frames = len(cv2_img_list)


    transformed_inputs = []
    for cv2_img in cv2_img_list:
        img_input = cv2Img_to_Image(cv2_img)
        img_input, _ = transforms(img_input, target=None)   # transforms 的最后有to_tensor 的操作
        
        transformed_inputs.append(img_input)
    
    total_predictions = []
    for fid in range(0,num_frames,batch_size):
        batch_inputs = transformed_inputs[fid:fid+batch_size]
        # batch_inputs = [img.to(model.device) for img in batch_inputs] # 在model 的forward里会有 to(self.device), stack 之后只需 to device 一次, 效率更高

        with torch.no_grad():
            batch_predictions = model(batch_inputs)  
        
        batch_predictions = [p.to(torch.device("cpu")).resize(_wh) for p in batch_predictions]  # 已经check过, resize 是 (width, height)
        total_predictions += batch_predictions
    
    assert len(total_predictions) == num_frames

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
    video_path = "datasets/vidvrd-dataset/videos/{}.mp4".format(video_name)
    assert os.path.isfile(video_path), \
        "Video: {} does not exist".format(video_path)

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

    cv2_img_list = VidRead2ImgNpyLits(video_path)
    results = detect_objects_on_single_video(model,transforms,cv2_img_list,args.batch_size)

    results4save = {video_name:results}
    save_path = "output/ILSVRC2015_train_00010018_det_results.json"
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

    assert args.part_id in [1,2,3]

    video_dir =  "datasets/vidvrd-dataset/videos"
    video_filenames = sorted(os.listdir(video_dir))
    video_paths_all = [os.path.join(video_dir,filename) for filename in video_filenames]
    if args.part_id == 1:
        video_paths = video_paths_all[:300]
    elif args.part_id == 2:
        video_paths = video_paths_all[300:600]
    elif args.part_id == 3:
        video_paths = video_paths_all[600:]

    output_dir = "output/VidVRD_det_results"

    
    
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

    for video_path in tqdm(video_paths):
        video_name = video_path.split('/')[-1].split('.')[0]
        save_path =  os.path.join(output_dir,"{}_det_results.json".format(video_name))
        if os.path.exists(save_path):
            print("{} exists, skip".format(save_path))
            continue
        
        cv2_img_list = VidRead2ImgNpyLits(video_path)
        results = detect_objects_on_single_video(model,transforms,cv2_img_list,args.batch_size)

        results4save = {video_name:results}
        with open(save_path,'w') as f:
            json.dump(results4save,f)
    
    print("Done.")


if __name__ == "__main__":

    # extract_signle_video_demo()
    main()
    '''
    python tools/extract_bboxes/extract_video_bboxes.py \
    --config_file tools/extract_bboxes/vinvl_x152c4_extrat_bbox.yaml \
    --batch_size 4 \
    MODEL.DEVICE cuda:2 \
    MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth \
    MODEL.ROI_HEADS.NMS_FILTER 1 \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
    TEST.IGNORE_BOX_REGRESSION False

    python tools/extract_bboxes/extract_video_bboxes.py \
    --config_file tools/extract_bboxes/vinvl_x152c4_extrat_bbox.yaml \
    --batch_size 2 \
    --part_id 3 \
    MODEL.DEVICE cuda:3 \
    MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth \
    MODEL.ROI_HEADS.NMS_FILTER 1 \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
    TEST.IGNORE_BOX_REGRESSION False
    '''
