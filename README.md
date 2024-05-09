# YoloV8 Detect and Track

### requirement:
***
```python
ultralytics==8.0.98

# val ------------------------------------------------------------------
pycocotools
tabulate

gitpython
tensorboard
numpy==1.23.1  # otherwise issues with track eval

# StrongSORT ------------------------------------------------------------------
easydict

# torchreid -------------------------------------------------------------------
gdown

# OCSORT ----------------------------------------------------------------------
filterpy

# Export ----------------------------------------------------------------------
onnx>=1.12.0  # ONNX export 
onnxsim>=0.4.1  # ONNX simplifier
nvidia-pyindex  # TensorRT export
nvidia-tensorrt  # TensorRT export
openvino-dev>=2022.3  # OpenVINO export
onnx2tf>=1.10.0


# Hyperparam search -----------------------------------------------------------
optuna
plotly  # for hp importance and pareto front plots
kaleido
joblib
```
&nbsp;
### Validation
***
##### 1. track.py
只能实现 `ocsort` 的跟踪算法，实现细节在 `./trackers` 目录
```commandline
mkdir ./valid_utils
cd ./valid_utils
mkdir ./data
```
`MOT17` 数据集存放到 `./valid_utils/data` 目录  
权重文件存放到 `./weight` 目录  
运行结果存放到 `./runs/val` 及 `./val_utils/data/trackers/mot_challenge` 目录

#### 2. examples/val.py
能实现 `'strongsort, ocsort', 'deepocsort`算法，实现细节在 `boxmot`目录
```shell
$ python3 val.py --tracking-method strongsort --benchmark MOT16
                 --tracking-method ocsort     --benchmark MOT17
                 --tracking-method ocsort     --benchmark <your-custom-dataset>
```
`MOT17` 数据集存放到 `./examples/val_utils/data` 目录  
权重文件存放到 `./examples/weights` 目录  
运行结果存放到 `./examples/runs/val` 及 `./examples/val_utils/data/trackers/mot_challenge` 目录

***
## 项目完善中ing...... 完善之后会整理出一个文档
***
&nbsp;
### Cite:
***
```
@software{Jocher_Ultralytics_YOLO_2023,
author = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
license = {AGPL-3.0},
month = jan,
title = {{Ultralytics YOLO}},
url = {https://github.com/ultralytics/ultralytics},
version = {8.0.0},
year = {2023}
}
```
```
@software{Brostrom_Real-time_multi-object_segmentation,
author = {Broström, Mikel},
doi = {https://zenodo.org/record/7629840},
license = {AGPL-3.0},
title = {{Real-time multi-object, segmentation and pose tracking using Yolov8 with DeepOCSORT and LightMBN}},
url = {https://github.com/mikel-brostrom/yolov8_tracking},
version = {8.0}
}
```