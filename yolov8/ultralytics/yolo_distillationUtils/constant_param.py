class opt:
    # YoloV8知识蒸馏
    yolo_kd_switch = False  # yolov8知识蒸馏开关
    teacher_model = "./teacher_model.pt"  # yolov8教师模型
    yolo_ratio = 0.25  # yolov8知识蒸馏比例
    yolo_kd_loss_selected = 'l2'  # 软标签损失的损失函数
    yolo_temperature = 20
    yolo_isL = False