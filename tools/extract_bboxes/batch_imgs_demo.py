# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 

import os.path as op
import argparse
import json

import cv2
import torch
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

def VidRead2ImgTensorLits(video_path):
    img_list = []
    cap = cv2.VideoCapture(video_path)  ##打开视频文件
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success = True
    count = 0
    while success and count < n_frames:
        success, image = cap.read()
        if success:
            image = torch.from_numpy(image).permute(2,0,1).float() # shape == (H,W,3) --> (3,H,W)
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
    # cv2_img = input_img
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img


def detect_objects_on_single_image(model, transforms, cv2_img):
    # cv2_img is the original input, so we can get the height and 
    # width information to scale the output boxes.
    img_input = cv2Img_to_Image(cv2_img)
    img_input, _ = transforms(img_input, target=None)
    print(img_input.shape)   # shape == (3,600,800)
    img_input = img_input.to(model.device)

    with torch.no_grad():
        prediction = model(img_input)
        prediction = prediction[0].to(torch.device("cpu"))

    img_height = cv2_img.shape[0]
    img_width = cv2_img.shape[1]


    prediction = prediction.resize((img_width, img_height))
    boxes = prediction.bbox.tolist()
    classes = prediction.get_field("labels").tolist()
    scores = prediction.get_field("scores").tolist()

    return [
        {"rect": box, "class": cls, "conf": score}
        for box, cls, score in
        zip(boxes, classes, scores)
    ]


def detect_objects_on_batch_imgs(model, transforms, cv2_imgs):
    # cv2_img is the original input, so we can get the height and 
    # width information to scale the output boxes.
    batch_inputs = []
    img_wh_list = []
    for cv2_img in cv2_imgs:
        h = cv2_img.shape[0]
        w = cv2_img.shape[1]
        img_wh_list.append((w,h))

        img_input = cv2Img_to_Image(cv2_img)
        img_input, _ = transforms(img_input, target=None)   # transforms 的最后有to_tensor 的操作
        print(img_input.shape)
        # img_input = img_input.to(model.device)  # 在model 的forward里会有 to(self.device), stack 之后只需 to device 一次, 效率更高

        batch_inputs.append(img_input)

    with torch.no_grad():
        batch_predictions = model(batch_inputs)  
        # 两个 image 的size 不一样，没关系
        # 在model 的forward里面的 to_image_list 里面会处理这个 stack 和padding的事情
        # 然后我们用在video里的时候， 多个帧组成一个batch，size都是一样的
        
        batch_predictions = [p.to(torch.device("cpu")) for p in batch_predictions]


    batch_results = []
    for prediction,(img_width, img_height) in zip(batch_predictions,img_wh_list):
        prediction = prediction.resize((img_width, img_height))
        boxes = prediction.bbox.tolist()  # len == num_bboxes
        classes = prediction.get_field("labels").tolist()
        scores = prediction.get_field("scores").tolist()

        batch_results.append(
            {"boxes": boxes, "classes": classes, "scores": scores}
        )

    return batch_results

def main():
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument("--config_file", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--labelmap_file", metavar="FILE",
                        help="labelmap file to select classes for visualizatioin")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    img_files = [
        "demo/xiaoshaonin.jpg",
        "demo/woman_fish.jpg",
    ]
    for img_file in img_files:
        assert op.isfile(img_file), \
            "Image: {} does not exist".format(img_file)

    # img_file = img_files[0]
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)

    save_files = [
        "output/xiaoshaonin_x152c4.batch2.jpg",
        "output/woman_fish_x152c4.batch2.jpg"
    ]

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

    

    # cv2_img = cv2.imread(img_file)
    # dets = detect_objects_on_single_image(model, transforms, cv2_img)
    cv2_imgs = [cv2.imread(img_file) for img_file in img_files]
    batch_results = detect_objects_on_batch_imgs(model,transforms,cv2_imgs)


    ## visualize
    for i, dets_per_img in enumerate(batch_results):
        boxes = dets_per_img["boxes"]
        classes =  [dataset_labelmap[c_id] for c_id in  dets_per_img["classes"]]
        scores = dets_per_img["scores"]

        draw_bb(cv2_imgs[i], boxes, classes, scores)

        save_file = save_files[i]
        cv2.imwrite(save_file, cv2_imgs[i])
        print("save results to: {}".format(save_file))

if __name__ == "__main__":
    main()
    '''
    python tools/extract_bboxes/batch_imgs_demo.py \
    --config_file tools/extract_bboxes/vinvl_x152c4_extrat_bbox.yaml \
    MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth \
    MODEL.ROI_HEADS.NMS_FILTER 1 \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
    TEST.IGNORE_BOX_REGRESSION False

    '''