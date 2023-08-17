# Code for extract bboxes with RoI features for VidVRD and VidOR dataset

we use [VinVL](https://github.com/pzzhang/VinVL) to extract traj bbox and RoI features.

we do this separately. The whole pipeline is 1)detect bbox --> 2)object tracking --> 3)extract RoI features. 
This is because at the tracking stage we will filter out many boxes, and extract RoI features based on the obtained object trajectory saves more computational cost.

We use `vinvl_vg_x152c4` as the pre-trained detector to do stages  1) & 3), and for stage 2) we use [Seq-NMS](https://github.com/tmoopenn/seq-nms) (which is parameter-free)

0. install VinVL and its Scene Graph Benchmark refer to https://github.com/pzzhang/VinVL

1. detect bbox:
    - For VidVRD, refer to `tools/extract_bboxes/extract_video_bboxes_dataloader.py`
    - For VidOR, refer to `tools/extract_bboxes/extract_video_bboxes_VideoReader.py`
    - For VidOR, beacuse the number of videos is very large (i.e., 7000), we use `VideoReader` from [decord](https://github.com/dmlc/decord) to load videos. It saves the memory and improves speed than using open-cv.

2. object tracking:
    - TODO

3. extract RoI features:
    - For VidVRD, refer to `tools/extract_features/extract_traj_features.py`
    - For VidOR, refer to `tools/extract_features/extract_traj_features_VideoReader.py`