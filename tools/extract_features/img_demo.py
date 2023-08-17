# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 

import cv2
import torch
import torch.nn.functional as F
import os.path as op
import argparse
import json
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

from tools.demo.visual_utils import draw_bb


def cv2Img_to_Image(input_img):
    cv2_img = input_img.copy()
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img

def detect_objects_on_single_image(model, transforms, cv2_img):
    # cv2_img is the original input, so we can get the height and 
    # width information to scale the output boxes.
    img_input = cv2Img_to_Image(cv2_img)
    img_input, _ = transforms(img_input, target=None)
    img_input = img_input.to(model.device)

    with torch.no_grad():
        prediction = model(img_input)
        prediction = prediction[0].to(torch.device("cpu"))

    img_height = cv2_img.shape[0]
    img_width = cv2_img.shape[1]


    prediction = prediction.resize((img_width, img_height))  # prediction is an object of `BoxList`, mode: "xyxy"
    # print(prediction.mode)
    boxes = prediction.bbox.tolist()
    classes = prediction.get_field("labels").tolist()
    scores = prediction.get_field("scores").tolist()
    box_features = prediction.get_field("box_features").tolist()

    return [
        {"rect": box, "class": cls, "conf": score, "box_features":feat}
        for box, cls, score, feat in
        zip(boxes, classes, scores,box_features)
    ]


def draft(model, transforms, cv2_img):
    # cv2_img is the original input, so we can get the height and 
    # width information to scale the output boxes.
    ori_height = cv2_img.shape[0]
    ori_width = cv2_img.shape[1]
    ori_wh = (ori_width,ori_height)
    img_input = cv2Img_to_Image(cv2_img)
    img_input, _ = transforms(img_input, target=None)
    input_h,input_w = img_input.shape[1],img_input.shape[2]
    input_wh = (input_w,input_h)
    

    ''' in AttrRCNN.forward
    images = to_image_list(images)
    images = images.to(self.device)
    features = self.backbone(images.tensors)

    proposals, proposal_losses = self.rpn(images, features, targets)
    x, predictions, detector_losses = self.roi_heads(features,proposals, targets)

    '''

    # 0 {'rect': [991.9342651367188, 1908.3780517578125, 3000.91748046875, 2725.4501953125], 'class': 'fish', 'conf': 0.9993058443069458}
    # 1 {'rect': [1977.78759765625, 898.5238037109375, 3707.15380859375, 2725.4501953125], 'class': 'shirt', 'conf': 0.8882750868797302}
    # 2 {'rect': [2338.898193359375, 285.08843994140625, 2973.72802734375, 952.0045166015625], 'class': 'head', 'conf': 0.8609465956687927}
    # 3 {'rect': [2317.74951171875, 285.910400390625, 3056.99609375, 663.0642700195312], 'class': 'hat', 'conf': 0.724172055721283}
    # 4 {'rect': [3133.188232421875, 1512.47265625, 4036.172119140625, 1752.4656982421875], 'class': 'handle', 'conf': 0.6738965511322021}
    bboxes = [
        [991.9342651367188, 1908.3780517578125, 3000.91748046875, 2725.4501953125],
        [1977.78759765625, 898.5238037109375, 3707.15380859375, 2725.4501953125],
        [2338.898193359375, 285.08843994140625, 2973.72802734375, 952.0045166015625],
        [2317.74951171875, 285.910400390625, 3056.99609375, 663.0642700195312],
        [3133.188232421875, 1512.47265625, 4036.172119140625, 1752.4656982421875]
    ]
    bboxes = BoxList(bboxes,ori_wh,mode='xyxy')
    bboxes = bboxes.resize(input_wh)



    with torch.no_grad():
        targets = None
        images = to_image_list(img_input)
        images = images.to(model.device)
        features = model.backbone(images.tensors)
        proposals, proposal_losses = model.rpn(images, features, targets)
        # proposals is a list of `BoxList` objects (mode=='xyxy'), with filed 'objectness', objectness 在 train RPN的时候用到，现在inference的时候用不到
        # len(proposals) == batch_size (i.e, number of imgs)
        # proposal_losses is an empty dict() because we are in test mode

        # feature_extractor generally corresponds to the pooler + heads  (这个heads里面是含有ResNet的一个stage的，应该是ResNet-C5)， 之前的backbone都是ResNet-C4
        # 所以说 feature_extractor 不止是在 feature map上crop一块然后做pooling
        # x = model.roi_heads.box.feature_extractor(features, proposals)
        
        # the type of feature_extractor is controlled by cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR, default: ResNet50Conv5ROIFeatureExtractor
        # 我们的 config sgg_configs/vgattr/vinvl_x152c4.yaml 中没有对其修改，所以用的就是默认的

        x, predictions, detector_losses = model.roi_heads(features, proposals, targets)
        
        print(type(proposals)) # list
        print(len(proposals),type(proposals[0]))  # len
        print(proposals[0].mode,proposals[0].fields())
        bbox = proposals[0].bbox  # w.r.t the resized image input (NOTE not normalized to 0~1), where the resize resolution is determined by `transforms`
        print(bbox,bbox.shape)  # shape == (300,4), 这个300是由 MODEL.RPN.POST_NMS_TOP_N_TEST 决定的

        assert False
        prediction = model(img_input)
        prediction = prediction[0].to(torch.device("cpu"))

    img_height = cv2_img.shape[0]
    img_width = cv2_img.shape[1]


    prediction = prediction.resize((img_width, img_height))
    boxes = prediction.bbox.tolist()
    classes = prediction.get_field("labels").tolist()
    scores = prediction.get_field("scores").tolist()
    features = prediction.get_field("box_features")
    print(features.shape)  # shape == (28,2048)
    
    
    

    return [
        {"rect": box, "class": cls, "conf": score}
        for box, cls, score in
        zip(boxes, classes, scores)
    ]



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
    print(ori_wh,input_wh)

    bboxes = BoxList(bboxes,ori_wh,mode='xyxy')
    bboxes = bboxes.resize(input_wh)

    ''' in AttrRCNN.forward
    images = to_image_list(images)
    images = images.to(self.device)
    features = self.backbone(images.tensors)

    proposals, proposal_losses = self.rpn(images, features, targets)
    x, predictions, detector_losses = self.roi_heads(features,proposals, targets)

    '''
    avg_pooler = torch.nn.AdaptiveAvgPool2d(1)

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

        # bbox_features = bbox_features.mean([2,3])  # (num_boxes, 2048)
        bbox_features = avg_pooler(bbox_features)    # (num_boxes, 2048, 1, 1)
        bbox_features = bbox_features.reshape(bbox_features.size(0),-1)
    
    print(bbox_features.shape)
    # assert False
    bbox_features = bbox_features.to(torch.device("cpu")).tolist()

    # box4print = bboxes.bbox.tolist()
    # for idx,box in enumerate(box4print):
    #     print(idx,box)
    
    return bbox_features



def extract_feature_given_bbox_debug(model, transforms, cv2_img, bboxes):
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

    print(ori_wh,input_wh)

    ''' in AttrRCNN.forward
    images = to_image_list(images)
    images = images.to(self.device)
    features = self.backbone(images.tensors)

    proposals, proposal_losses = self.rpn(images, features, targets)
    x, predictions, detector_losses = self.roi_heads(features,proposals, targets)

    '''

    with torch.no_grad():
        images = to_image_list(img_input)
        images = images.to(model.device)
        # bboxes = bboxes.to(model.device)
        features = model.backbone(images.tensors) 
        # features: list[tensor], len == 1 for ResNet-C4 backbone, for FPN, len == num_levels
        # features[0].shape == (batch_size, 1024, H/16, W/16), (W,H) == input_wh

        proposals, proposal_losses = model.rpn(images, features, None)
        bbox_features = model.roi_heads.box.feature_extractor(features, proposals)  # 
        # print(bbox_features.shape)
        features_before = bbox_features.clone().mean([2,3])
        class_logits, box_regression = model.roi_heads.box.predictor(bbox_features)

        result = model.roi_heads.box.post_processor((class_logits, box_regression),proposals, bbox_features)
        result = result[0]
        proposals = proposals[0].to(torch.device("cpu"))
        print(result)
    prediction = result
    prediction = prediction.resize(ori_wh)
    boxes = prediction.bbox.tolist()
    classes = prediction.get_field("labels").tolist()
    scores = prediction.get_field("scores").tolist()
    features = prediction.get_field("box_features")
    
    assert features.shape == (28,2048)
    print(features.shape,features_before.shape)  # (28,2048); (300, 2048)
    for idx,score in enumerate(scores):
        print(idx,score,features[idx,:10].tolist())
    

    dist = torch.abs(features[None,:,:] - features_before[:,None,:])  # (300,28,2048)
    dist = dist.sum(dim=-1)
    mask = dist < 1.0
    assert mask.sum() == features.shape[0]
    row_ids,col_ids = mask.nonzero(as_tuple=True)
    # print(dist.min(),mask.sum())
    print("=-"*20)

    for idx in row_ids.tolist():
        print(idx,proposals.bbox[idx,:].tolist(),features_before[idx,:10].tolist())
    
    return 0


def ImgOnly_demo():
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument("--config_file", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--img_file", metavar="FILE", help="image path")
    parser.add_argument("--labelmap_file", metavar="FILE",
                        help="labelmap file to select classes for visualizatioin")
    parser.add_argument("--save_file", required=False, type=str, default=None,
                        help="filename to save the proceed image")
    parser.add_argument("--visualize_attr", action="store_true",   
    # 'store_true' 代表运行程序时加上 --visualize_attr， 就代表 visualize_attr取True， 不加的时候就默认False

                        help="visualize the object attributes")
    parser.add_argument("--visualize_relation", action="store_true",
                        help="visualize the relationships")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    assert op.isfile(args.img_file), \
        "Image: {} does not exist".format(args.img_file)

    output_dir = cfg.OUTPUT_DIR
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
    dataset_labelmap = {int(val): key for key, val in dataset_allmap['label_to_idx'].items()}


    transforms = build_transforms(cfg, is_train=False)
    cv2_img = cv2.imread(args.img_file)
    dets = detect_objects_on_single_image(model, transforms, cv2_img)


    for obj in dets:
        obj["class"] = dataset_labelmap[obj["class"]]
    
    for idx, obj in enumerate(dets):
        box_features = obj["box_features"][:10]
        print(idx,obj["rect"],box_features)
        # print(idx,obj["rect"])
        
        '''
        0 [0.0, 0.0, 0.0, 0.9847757816314697, 0.0, 0.0, 0.0, 0.17932623624801636, 0.0, 0.9580087065696716]
        1 [0.0, 0.0025060127954930067, 0.08155769854784012, 1.5867893695831299, 0.04989328607916832, 0.10120873153209686, 0.0, 2.977135181427002, 0.35753706097602844, 0.0]
        2 [1.3723793029785156, 0.597181499004364, 0.0, 0.5496529936790466, 0.9378848075866699, 0.0, 0.35644352436065674, 0.0, 0.5265184044837952, 0.0]
        3 [0.20207169651985168, 1.2638163566589355, 0.0, 0.04261248931288719, 0.6497355699539185, 0.0, 0.5092315077781677, 0.0, 0.07531891763210297, 0.0]
        4 [0.0, 0.0, 0.013461998663842678, 0.0, 0.00171277963090688, 0.0, 0.03787873685359955, 1.4367340803146362, 0.0, 0.0]
        5 [0.2640811800956726, 0.0, 0.14080046117305756, 1.8049826622009277, 0.6606996655464172, 0.0, 0.03810497000813484, 12.52658462524414, 0.0, 0.010729575529694557]
        '''


    rects = [d["rect"] for d in dets]
    scores = [d["conf"] for d in dets]
    labels = [d["class"] for d in dets]

    draw_bb(cv2_img, rects, labels, scores)


    if not args.save_file:
        save_file = op.splitext(args.img_file)[0] + ".detect.jpg"
    else:
        save_file = args.save_file
    cv2.imwrite(save_file, cv2_img)
    print("save results to: {}".format(save_file))


def ImgGivenBox_demo():
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument("--config_file", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--img_file", metavar="FILE", help="image path")
    parser.add_argument("--labelmap_file", metavar="FILE",
                        help="labelmap file to select classes for visualizatioin")
    parser.add_argument("--save_file", required=False, type=str, default=None,
                        help="filename to save the proceed image")
    parser.add_argument("--visualize_attr", action="store_true",   # 'store_true' 代表运行程序时加上 --visualize_attr， 就代表 visualize_attr取True， 不加的时候就默认False
                        help="visualize the object attributes")
    parser.add_argument("--visualize_relation", action="store_true",
                        help="visualize the relationships")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    assert op.isfile(args.img_file), \
        "Image: {} does not exist".format(args.img_file)

    output_dir = cfg.OUTPUT_DIR
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
    dataset_labelmap = {int(val): key for key, val in dataset_allmap['label_to_idx'].items()}


    transforms = build_transforms(cfg, is_train=False)
    cv2_img = cv2.imread(args.img_file)

    # 0 {'rect': [991.9342651367188, 1908.3780517578125, 3000.91748046875, 2725.4501953125], 'class': 'fish', 'conf': 0.9993058443069458}
    # 1 {'rect': [1977.78759765625, 898.5238037109375, 3707.15380859375, 2725.4501953125], 'class': 'shirt', 'conf': 0.8882750868797302}
    # 2 {'rect': [2338.898193359375, 285.08843994140625, 2973.72802734375, 952.0045166015625], 'class': 'head', 'conf': 0.8609465956687927}
    # 3 {'rect': [2317.74951171875, 285.910400390625, 3056.99609375, 663.0642700195312], 'class': 'hat', 'conf': 0.724172055721283}
    # 4 {'rect': [3133.188232421875, 1512.47265625, 4036.172119140625, 1752.4656982421875], 'class': 'handle', 'conf': 0.6738965511322021}
    bboxes_list = [
        [991.9342651367188, 1908.3780517578125, 3000.91748046875, 2725.4501953125],
        [1977.78759765625, 898.5238037109375, 3707.15380859375, 2725.4501953125],
        [2338.898193359375, 285.08843994140625, 2973.72802734375, 952.0045166015625],
        [2317.74951171875, 285.910400390625, 3056.99609375, 663.0642700195312],
        [3133.188232421875, 1512.47265625, 4036.172119140625, 1752.4656982421875]
    ]
    

    box_features = extract_feature_given_bbox(model, transforms, cv2_img, bboxes_list)


    for idx,box_feature in enumerate(box_features):
        print(idx,box_feature[:10])

        '''
        0 [0.0, 0.0, 0.0, 1.1339911222457886, 0.0, 0.0, 0.04108462110161781, 0.9963603019714355, 0.0, 0.9344449639320374]
        1 [0.09081020951271057, 0.0, 0.02639104798436165, 1.8454937934875488, 0.0, 0.05617659166455269, 0.4271577298641205, 6.5853681564331055, 0.4914681613445282, 0.4103371798992157]
        2 [1.1863552331924438, 0.9544329047203064, 0.0, 0.9082130789756775, 0.8100321292877197, 0.0, 0.4404435455799103, 0.2569000720977783, 0.8241412043571472, 0.0]
        3 [0.21225278079509735, 1.2694249153137207, 0.0, 0.1301029771566391, 0.634881854057312, 0.0, 0.4620363712310791, 0.0, 0.1437022089958191, 0.0]
        4 [0.0, 0.0, 0.0, 0.0, 0.00392921594902873, 0.0, 0.0, 1.5948925018310547, 0.0, 0.0]
        '''


if __name__ == "__main__":
    # main()
    # ImgOnly_demo()
    ImgGivenBox_demo()
    # 这两个输出的 RoI feature不一样，这是因为对feature_extractor输入的bbox不一样
    # 在ImgOnly_demo()中这个bbox是 rpn的proposal， 
    # 但是在ImgGivenBox_demo()中， 这个box是最终预测的box，与rpn给出的box有查别，但是比rpn的box更准一点，
    # 所以ImgGivenBox_demo()输出的roi feature 没有问题，甚至更好

    '''
    python tools/extract_features/img_demo.py \
    --config_file sgg_configs/vgattr/vinvl_x152c4.yaml \
    --img_file demo/woman_fish.jpg \
    --save_file output/woman_fish_x152c4.obj2.jpg \
    MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth \
    MODEL.ROI_HEADS.NMS_FILTER 1 \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
    TEST.IGNORE_BOX_REGRESSION False \
    TEST.OUTPUT_FEATURE True
    '''
