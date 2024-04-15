import torchreid
from multiprocessing import freeze_support
import torch

from ReIDNet.distillation_utils import build_model
from ReIDNet.distillation_utils.datamanager import ImageDataManager
from ReIDNet.distillation_utils.softmax import ImageSoftmaxEngine

if __name__ == '__main__':
    freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamanager = ImageDataManager(
        root='reid-data',  # path to market1501
        sources='market1501',
        height=256,
        width=128,
        batch_size_test=32,
        batch_size_train=100,
        market1501_500k=False,
        combineall=True
    )
    student_model = build_model(
        name="osnet_x0_25",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True
    )

    teacher_model = build_model(
        name="osnet_x1_0",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True
    )

    student_model.to(device)
    teacher_model.to(device)
    optimizer = torchreid.optim.build_optimizer(
        student_model,
        optim="sgd",
        lr=0.01,
        staged_lr=True,
        new_layers='classifier',
        base_lr_mult=0.1
    )
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=20
    )
    engine = ImageSoftmaxEngine(
        datamanager,
        student_model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )
    engine.run(
        save_dir="log/osnet",
        max_epoch=300,
        eval_freq=10,
        print_freq=10,
        fixbase_epoch=5,
        open_layers='classifier',
        teacher_model=teacher_model,
        kd=True
    )
