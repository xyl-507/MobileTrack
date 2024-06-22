# [IET Image Processing 2022] MobileTrack: Siamese efficient mobile network for high-speed UAV tracking

This is an official pytorch implementation of the 2022 IET Image Processing paper: 
```
MobileTrack: Siamese efficient mobile network for high-speed UAV tracking
(accepted by IET Image Processing, DOI: 10.1049/ipr2.12565)
```

![image](https://github.com/xyl-507/MobileTrack/blob/main/figs/fig.jpg)

The paper can be downloaded from [IET Image Processing](https://doi.org/10.1049/ipr2.12565)

The models and raw results can be downloaded from [BaiduYun](https://pan.baidu.com/s/1qyQmZg12Kd9J2Bc3BMX-cQ?pwd=1234). 

### UAV Tracking

| Datasets | mobiletrack_r50_l234|
| :--------------------: | :----------------: |
| UAV123(Suc./Pre.) | 0.609/0.813|
| UAVDT(Suc./Pre.) | 0.559/0.774|
| DTB70(Suc./Pre.) | 0.612/0.814 |

Note:

-  `r50_lxyz` denotes the outputs of stage x, y, and z in [ResNet-50](https://arxiv.org/abs/1512.03385).
- The suffixes `DTB70` is designed for the DTB70, the default (without suffix) is designed for UAV20L and UAVDT.
- `e20` in parentheses means checkpoint_e20.pth

## Installation

Please find installation instructions in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using MobileTrack

### Add SmallTrack to your PYTHONPATH

```bash
export PYTHONPATH=/path/to/mobiletrack:$PYTHONPATH
```


### demo

```bash
python tools/demo.py \
    --config experiments/siamban_mobilev2_l234/config.yaml \
    --snapshot experiments/siamban_mobilev2_l234/MobileTrack.pth
    --video demo/bag.avi
```

### Download testing datasets

Download datasets and put them into `testing_dataset` directory. Jsons of commonly used datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI) or [BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F). If you want to test tracker on new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to setting `testing_dataset`. 

### Test tracker

```bash
cd experiments/siamban_mobilev2_l234
python -u ../../tools/test.py 	\
	--snapshot MobileTrack.pth 	\ # model path
	--dataset UAV123 	\ # dataset name
	--config config.yaml	  # config file
```

The testing results will in the current directory(results/dataset/model_name/)

### Eval tracker

assume still in experiments/siamban_mobilev2_l234

``` bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset UAV123         \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'ch*'   # tracker_name
```

###  Training :wrench:

See [TRAIN.md](TRAIN.md) for detailed instruction.


### Acknowledgement
The code based on the [PySOT](https://github.com/STVIR/pysot) , [SiamBAN](https://github.com/hqucv/siamban) ,
[MobileNetV2](https://ieeexplore.ieee.org/abstract/document/8578572) and [ECA-Net](https://ieeexplore.ieee.org/document/9156697)
We would like to express our sincere thanks to the contributors.

### Citation:
If you find this work useful for your research, please cite the following papers:
```
@article{https://doi.org/10.1049/ipr2.12565,
author = {Xue, Yuanliang and Jin, Guodong and Shen, Tao and Tan, Lining and Yang, Jing and Hou, Xiaohan},
title = {MobileTrack: Siamese efficient mobile network for high-speed UAV tracking},
journal = {IET Image Processing},
volume = {16},
number = {12},
pages = {3300-3313},
doi = {https://doi.org/10.1049/ipr2.12565},
url = {https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/ipr2.12565},
eprint = {https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/ipr2.12565},
abstract = {Abstract Recently, Siamese-based trackers have drawn amounts of attention in visual tracking field because of their excellent performance. However, visual object tracking on Unmanned Aerial Vehicles platform encounters difficulties under circumstances such as small objects and similar objects interference. Most existing tracking methods for aerial tracking adopt deeper networks or inefficient policies to promote performance, but most trackers can hardly meet real-time requirements on mobile platforms with limited computing resources. Thus, in this work, an efficient and lightweight siamese tracker (MobileTrack) is proposed for high-time Unmanned Aerial Vehicles tracking, realising the balance between performance and speed. Firstly, a lightweight convolutional network (D-MobileNet) is designed to enhance the characterisation ability of small objects. Secondly, an efficient object-aware module is proposed for local cross-channel information exchange, enhancing the feature information of the tracking object. Besides, an anchor-free region proposal network is introduced to predict the object pixel by pixel. Finally, deep and shallow feature information is fully utilised by cascading multiple anchor-free region proposal networks for accurate locating and robust tracking. Extensive experiments on the three Unmanned Aerial Vehicles benchmarks show that the proposed tracker achieves outstanding performance while keeping a beyond-real-time speed.},
year = {2022}
}
```
If you have any questions about this work, please contact with me via xyl_507@outlook.com
