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

def move_det_results():
    src_dir = "output/VidVRD_det_results_xsn"
    dst_dir = "output/VidVRD_det_results"

    scr_files = sorted(os.listdir(src_dir)) #[ for filename in  ]
    dst_files = sorted(os.listdir(dst_dir)) # [os.path.join(dst_dir,filename) for filename in  ]

    for x in tqdm(scr_files):
        if x in dst_files:
            continue

        src_path = os.path.join(src_dir,x)
        dst_path = os.path.join(dst_dir,x)
        cmd = "cp {} {}".format(src_path,dst_path)
        os.system(cmd)
        # print(cmd)
        time.sleep(0.1)

def assert_len_demo():
    xx =  "/home/gkf/project/scene_graph_benchmark/output/VidVRD_det_results"
    det_names = sorted(os.listdir(xx))
    zz = "datasets/vidvrd-dataset/images"
    video_names = sorted(os.listdir(zz))
    print(len(video_names),len(det_names))
    for d,v in zip(det_names,video_names):
        frame_names = os.listdir(os.path.join(zz,v))
        num_frames = len(frame_names)
        # print(v)
        # assert False
        res_path = os.path.join(xx,d)
        with open(res_path,'r') as f:
            dets = json.load(f)
        num_dets = len(dets[v])

        # print(v,num_frames,num_dets)
        # break
        assert num_frames == num_dets

def demo2():
    xx = "output/VidORval_det_results"
    len_ = len(os.listdir(xx))
    print(len_)

if __name__ == "__main__":
    pass
    # test_tqdm()
    # move_det_results()
    demo2()
    # assert_len_demo()