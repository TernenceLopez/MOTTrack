# YoloV8 Detect and Track

### requirement:
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

## 项目完善中ing...... 完善之后会整理出一个文档