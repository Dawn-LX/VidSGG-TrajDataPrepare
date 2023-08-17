
import os
import argparse
import json
from tqdm import tqdm
from decord import VideoReader

import cv2
import numpy as np
import torch

from PIL import Image


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

def NpyImg_to_Image(input_img):
    # the channel order of input_img is RGB
    img = Image.fromarray(input_img)
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
        if self.transforms is not None:
            img, _ = self.transforms(img, target=None)   # transforms 的最后有to_tensor 的操作

        return wh,img

class Dataset_VidVRD_v2(object):
    def __init__(self,video_path,transforms):
        super().__init__()

        self.video_reader =  VideoReader(video_path,)
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
        if self.transforms is not None:
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


def detect_objects_on_single_video(frames_dir):
    # cv2_img is the original input, so we can get the height and 
    # width information to scale the output boxes.
    video_name = frames_dir.split('/')[-1]

    dataset = Dataset_VidVRD(frames_dir,None)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        collate_fn = collator_func,
        num_workers = 2,
        drop_last=False,
        shuffle=False,
    )
    dataset_len = len(dataset)
    total_predictions = []
    for wh,batch_imgs in tqdm(dataloader,position=1, desc="{}".format(video_name), leave=False, ncols=160):
        pass
    
    # assert len(total_predictions) == dataset_len

    return total_predictions


def detect_objects_on_single_video_v2(video_path):
    # cv2_img is the original input, so we can get the height and 
    # width information to scale the output boxes.
    video_name = video_path.split('/')[-1].split('.')[0]

    dataset = Dataset_VidVRD_v2(video_path,None)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        collate_fn = collator_func,
        num_workers = 0,
        drop_last=False,
        shuffle=False,
    )
    dataset_len = len(dataset)
    total_predictions = []
    for wh,batch_imgs in tqdm(dataloader,position=1, desc="{}".format(video_name), leave=False, ncols=160):
        pass
    
    # assert len(total_predictions) == dataset_len

    return total_predictions

def main():
    

    video_dir =  "datasets/vidvrd-dataset/images"
    video_names = sorted(os.listdir(video_dir))
    frames_dir_all = [os.path.join(video_dir,name) for name in video_names]
    frames_dirs = frames_dir_all


    for frames_dir_per_video in tqdm(frames_dirs, position=0, desc="all videos (use cv2.imread)", leave=False, ncols=160):

        
        detect_objects_on_single_video(frames_dir_per_video)

    

    print("Done.")


def main_use_VideoReader():
    
    video_dir =  "datasets/vidvrd-dataset/videos"
    output_dir = "output/VidVRD_det_results"
    video_filenames = sorted(os.listdir(video_dir))
    video_paths = [os.path.join(video_dir,filename) for filename in video_filenames]


    for video_path in tqdm(video_paths, position=0, desc="all videos (use VideoReader)", leave=False, ncols=160):

        
        detect_objects_on_single_video_v2(video_path)



def VideoReader_demo():
    pass
    video_dir =  "datasets/vidvrd-dataset/videos"
    output_dir = "output/VidVRD_det_results"
    video_filenames = sorted(os.listdir(video_dir))
    
    video_name = "ILSVRC2015_train_00005005"

    video_path = os.path.join(video_dir,video_name+'.mp4')
    video_reader =  VideoReader(video_path)
    print(len(video_reader))
    frames_all = []
    cv2_frames_all = []
    for i in range(len(video_reader)):
        frame = video_reader[i].asnumpy()
        # frame = video_reader.get_batch([i]).asnumpy().squeeze(0)
        # print(frame.shape,type(frame))
        frame_path = "datasets/vidvrd-dataset/images/{}/{:06d}.JPEG".format(video_name,i)
        cv2_frame = cv2.imread(frame_path)
        cv2_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        assert frame.shape == cv2_frame.shape
        h,w,_ = frame.shape
        # assert (frame == cv2_frame).sum() == h*w*3  # not identical with cv2_img, but this is fine
        
        # eq_sum = (frame == cv2_frame).sum()
        # total_numel = h*w*3
        # diff_abs = np.abs(frame - cv2_frame)
        # print(frame[1,1,:])
        # print(cv2_frame[1,1,:])
        # print(diff_abs.max())
        # print(eq_sum,total_numel,eq_sum/total_numel)
        # break
        frames_all.append(frame)
        cv2_frames_all.append(cv2_frame)

    frames_all = np.stack(frames_all,axis=0)
    cv2_frames_all = np.stack(cv2_frames_all,axis=0)
    np.save("tools/demo/frames_all.npy",frames_all)
    np.save("tools/demo/cv2_frames_all.npy",frames_all)


def API_demo():
    from collections import defaultdict
    
    n_trajs = [2,4,5,1,4,5]
    xx = torch.randn(size=(sum(n_trajs),3))
    xx = torch.split(xx,n_trajs,dim=0)
    for x in xx:
        print(x.shape)
    
    global_id = 0
    tids = defaultdict(list)
    for x in xx:
        n = x.shape[0]
        tids[n].append(global_id)
        global_id += 1
    print(tids)

        

if __name__ == "__main__":
    
    # main()
    # main_use_VideoReader()
    # VideoReader_demo()
    API_demo()
