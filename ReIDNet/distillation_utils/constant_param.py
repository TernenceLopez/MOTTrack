class opt:
    ratio = 0.25
    kd_loss_selected = 'l2'  # 软标签损失的损失函数
    temperature = 20
    isL = False
    yolo_kd_switch = True  # yolov8知识蒸馏开关
    teacher_model = "./teacher_model.pt"
