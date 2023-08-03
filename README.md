# DO-SAR

Official implementation of the paper 'DO-SA&R: Distant Object Augmented Set Abstraction and Regression for Point-based 3D Object Detection'.
        

![image](https://github.com/mikasa3lili/DO-SAR/blob/main/docs/pipeline.png)

# Abstract

Point-based 3D detection approaches usually suffer from the severe point sampling imbalance problem between foreground and background. We observe that prior works have attempted to alleviate this imbalance by emphasizing foreground sampling. However, even adequate foreground sampling may be extremely unbalanced between nearby and distant objects, yielding unsatisfactory performance in detecting distant objects. To tackle this issue, this paper first proposes a novel method named Distant Object Augmented Set Abstraction and Regression (DO-SA&R) to enhance distant object detection, which is vital for the timely response of decision-making systems like autonomous driving. Technically, our approach first designs DO-SA with novel distant object augmented farthest point sampling (DO-FPS) to emphasize sampling on distant objects by leveraging both object-dependent and depth-dependent information. Then, we propose distant object augmented regression to reweight all the instance boxes for strengthening regression training on distant objects. In practice, the proposed DO-SA&R can be easily embedded into the existing modules, yielding consistent performance improvements, especially on detecting distant objects. Extensive experiments are conducted on the popular KITTI, nuScenes and Waymo datasets, and DO-SA&R demonstrates superior performance, especially for distant object detection. 

# Requirements
(Our code is tested on:)  
Pyton 3.8  
Pytorch 1.7.1+cu110  
Spconv v1.2.1  
Cuda 11.0  

# Installation
a. Clone this repository:   
`git clone https://github.com/mikasa3lili/DO-SAR`  
`cd DO-SAR`  

b. Install spconv v1.2.1:  
`git clone https://github.com/traveller59/spconv.git`  
`cd spconv`  
`git checkout v1.2.1`  
`git submodule update --init --recursive`  
`python setup.py bdist_wheel`  
`pip install ./dist/spconv-1.2.1-cp36-cp36m-linux_x86_64.whl`    
`cd ..`  

c. Install pcdet toolbox:  
`pip install -r requirements.txt`  
`python setup.py develop`  

# Data Preparation  
You shoud download the KITTI, nuScenes, Waymo datasets, and follow the OpenPCDet(https://github.com/open-mmlab/OpenPCDet) to generate data infos.  
These datasets shold have the following organization:  
KITTI:  
`├── data`  
`│   ├── kitti`  
`│   │   │── ImageSets`  
`│   │   │── training`  
`│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)`  
`│   │   │── testing`  
`│   │   │   ├──calib & velodyne & image_2`  
nuScenes:  
`├── data`  
`│   ├── nuscenes`  
`│   │   │── v1.0-trainval (or v1.0-mini if you use mini)`  
`│   │   │   │── samples`  
`│   │   │   │── sweeps`  
`│   │   │   │── maps`  
`│   │   │   │── v1.0-trainval`  
Waymo:  
`│   ├── waymo`  
`│   │   │── ImageSets`  
`│   │   │── raw_data`  
`│   │   │   │── segment-xxxxxxxx.tfrecord`  
`|   |   |   |── ...`  
`|   |   |── waymo_processed_data_v0_5_0`  
`│   │   │   │── segment-xxxxxxxx/`  
`|   |   |   |── ...`  
`│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1/  (old, for single-frame)`  
`│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl  (old, for single-frame)`  
`│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy (optional, old, for single-frame)`  
`│   │   │── waymo_processed_data_v0_5_0_infos_train.pkl (optional)`  
`│   │   │── waymo_processed_data_v0_5_0_infos_val.pkl (optional)`  
`|   |   |── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_multiframe_-4_to_0 (new, for single/multi-frame)`  
`│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1_multiframe_-4_to_0.pkl (new, for single/multi-frame)`  
`│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_multiframe_-4_to_0_global.np  (new, for single/multi-frame)`  
# Training and Testing  
python train.py --cfg_file ${CONFIG_FILE} (--ckpt ${CKPT})  
python test.py --cfg_file ${CONFIG_FILE} --ckpt ${CKPT}  

train with multiple gpus:  
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}  

# Acknowledgement
Our project is developed based on [SASA](https://github.com/blakechen97/SASA). Thanks for this excellence work!
