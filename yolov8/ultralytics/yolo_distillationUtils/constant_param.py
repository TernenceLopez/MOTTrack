class opt:
    # YoloV8知识蒸馏
    yolo_kd_switch = True  # yolov8知识蒸馏开关
    teacher_model = "./teacher_model.pt"  # yolov8教师模型
    yolo_ratio = 0.25  # yolov8知识蒸馏比例
    yolo_kd_loss_selected = 'l2'  # 软标签损失的损失函数
    yolo_temperature = 20
    yolo_AdaptiveParams = False  # 开启知识蒸馏自适应参数(需要准备一份preTrain的学生模型才能开启，在default.yaml中设置)
    yolo_isL = False
    # 知识蒸馏各部分损失权重
    yolo_hard_loss_weight = 2
    yolo_soft_loss_weight = 1
    yolo_attention_loss_weight = 1.5
    # 知识蒸馏损失记录
    xlsx_file_name = "./loss_record.xlsx"
