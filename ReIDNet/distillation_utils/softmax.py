from __future__ import division, print_function, absolute_import

from torch import nn
from torchreid.reid import metrics
from torchreid.reid.losses import CrossEntropyLoss

from ReIDNet.distillation_utils.constant_param import opt
from ReIDNet.distillation_utils.engine import Engine
from ReIDNet.distillation_utils.experimental import get_s_feas_by_hook, get_t_feas_by_hook
from ReIDNet.distillation_utils.kd_loss import compute_kd_output_loss, EFKD
from ReIDNet.kd_losses.st import SoftTarget


# from torchreid.reid.engine.engine import Engine


class ImageSoftmaxEngine(Engine):
    r"""Softmax-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::

        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501',
            print_freq=10
        )
    """

    def __init__(
            self,
            datamanager,
            model,
            optimizer,
            scheduler=None,
            use_gpu=True,
            label_smooth=True
    ):
        super(ImageSoftmaxEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.feature_loss = EFKD(opt.isL)  # 特征图转移损失
        self.soft_loss = SoftTarget(opt.temperature)  # 软标签损失
        # self.sigmod_fun = nn.Sigmoid()
        # self.sigmod_fun_2 = nn.Sigmoid()

    def forward_backward(self, data, teacher_model, kd):
        # 子类重写父类方法
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        student_pred = 0
        loss = 0
        if not kd:  # 不进行知识蒸馏
            student_pred = self.model(imgs)
            loss = self.compute_loss(self.criterion, outputs, pids)
        else:  # 进行知识蒸馏
            s_f = get_s_feas_by_hook(self.model)  # hook机制
            student_pred = self.model(imgs)  # forward
            t_f = get_t_feas_by_hook(teacher_model)
            teacher_pred = teacher_model(imgs)
            loss_hard = self.compute_loss(self.criterion, student_pred, pids)  # 硬标签损失
            ftloss, ftloss_items = self.feature_loss(pids.cuda(), t_f, s_f, opt.ratio)  # 特征图转移损失
            kdloss = self.soft_loss(student_pred, teacher_pred)  # 软标签损失
            # theta = self.sigmod_fun(theta)
            # beta = self.sigmod_fun_2(beta)
            # loss = loss_hard + beta * kdloss + theta * ftloss
            loss = loss_hard + kdloss + ftloss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_summary = {
            'loss': loss.item(),
            'acc': metrics.accuracy(student_pred, pids)[0].item()
        }

        return loss_summary
