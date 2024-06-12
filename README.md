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
