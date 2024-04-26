class opt:
    # ReID Net知识蒸馏
    ratio = 0.25
    kd_loss_selected = 'l2'  # 软标签损失的损失函数
    temperature = 20
    isL = False
