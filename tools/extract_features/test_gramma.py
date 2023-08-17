# import numpy as np
import torch
import os
from tqdm import tqdm
import time
import json

def test_batchsize():
    inputs = torch.randint(9,88,size=(4,)).tolist()

    print(inputs)


    batch_size = 4

    num_frames = len(inputs)

    for fid in range(0,num_frames,batch_size):
        batch_inputs = inputs[fid:fid+batch_size]

        print(fid,batch_inputs)

def test_tqdm():
    jj = [7,7,3,1,2,3,4,5]*20
    for i in tqdm(range(len(jj)), position=0, desc="all videos", leave=False, ncols=160):
        for j in tqdm(range(jj[i]), position=1, desc="video_{:04d}".format(i), leave=False, ncols=160):
            time.sleep(0.5)
    
    # for i in tqdm(range(len(jj)), position=0, desc="all videos", leave=False):
    #     for j in tqdm(range(jj[i]), position=1, desc="video_{:04d}".format(i), leave=False):
    #         time.sleep(0.5)

def demo():
    xx =  "/home/gkf/project/scene_graph_benchmark/output/VidVRD_det_results"
    det_names = sorted(os.listdir(xx))
    zz = "datasets/vidvrd-dataset/images"
    video_names = sorted(os.listdir(zz))
    print(len(video_names),len(det_names))
    for d,v in zip(det_names,video_names):
        # pass
        v = v + "_det_results.json"
        # break
        assert d == v

def len_demo():
    tracking_res_dir = "/home/gkf/project/VidVRD-II/tracklets_results/VidVRD_segment_tracking_results/"
    tracking_res_names = sorted(os.listdir(tracking_res_dir))
    print(len(tracking_res_names))

if __name__ == "__main__":
    pass
    # test_tqdm()
    # move_det_results()
    # demo()
    len_demo()
    '''
    The output feature will be encoded as base64

    # extract vision features with VinVL object-attribute detection model

    python tools/test_sg_net.py \
        --config-file sgg_configs/vgattr/vinvl_x152c4.yaml \
        TEST.IMS_PER_BATCH 2 \
        MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth \
        MODEL.ROI_HEADS.NMS_FILTER 1 \
        MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
        DATA_DIR "../maskrcnn-benchmark-1/datasets1" \
        TEST.IGNORE_BOX_REGRESSION True \
        MODEL.ATTRIBUTE_ON True
    
    To extract relation features (union bounding box's feature), in yaml file, set TEST.OUTPUT_RELATION_FEATURE to True, add 'relation_feature' in TEST.TSV_SAVE_SUBSET.

    To extract bounding box features, in yaml file, set TEST.OUTPUT_FEATURE to True, add 'feature' in TEST.TSV_SAVE_SUBSET.
    '''