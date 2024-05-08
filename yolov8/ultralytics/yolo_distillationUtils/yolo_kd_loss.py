import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov8.ultralytics.yolo.utils.loss import BboxLoss
from yolov8.ultralytics.yolo.utils.ops import xywh2xyxy
from yolov8.ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors


def compute_kd_output_loss(pred, teacher_pred, model, kd_loss_selected="l2", temperature=20, reg_norm=None):
    t_ft = torch.cuda.FloatTensor if teacher_pred[0].is_cuda else torch.Tensor
    t_lcls, t_lbox, t_lobj = t_ft([0]), t_ft([0]), t_ft([0])
    h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)
    if red != "mean":
        raise NotImplementedError(
            "reduction must be mean in distillation mode!")

    KDboxLoss = nn.MSELoss(reduction="none")
    if kd_loss_selected == "l2":
        KDclsLoss = nn.MSELoss(reduction="none")
    elif kd_loss_selected == "kl":
        KDclsLoss = nn.KLDivLoss(reduction="none")
    else:
        KDclsLoss = nn.BCEWithLogitsLoss(reduction="none")
    KDobjLoss = nn.MSELoss(reduction="none")
    # per output
    for i, pi in enumerate(pred):  # layer index, layer predictions
        # t_pi  -->  torch.Size([16, 3, 80, 80, 25])
        t_pi = teacher_pred[i]
        # t_obj_scale  --> torch.Size([16, 3, 80, 80])
        t_obj_scale = t_pi[..., 4].sigmoid()
        # zero = torch.zeros_like(t_obj_scale)
        # t_obj_scale = torch.where(t_obj_scale < 0.5, zero, t_obj_scale)

        # BBox
        # repeat 是沿着原来的维度做复制  torch.Size([16, 3, 80, 80, 4])
        b_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1, 1, 1, 1, 4)
        if not reg_norm:
            t_lbox += torch.mean(KDboxLoss(pi[..., :4], t_pi[..., :4]) * b_obj_scale)
        else:
            wh_norm_scale = reg_norm[i].unsqueeze(0).unsqueeze(-2).unsqueeze(-2)
            # pxy
            t_lbox += torch.mean(KDboxLoss(pi[..., :2].sigmoid(), t_pi[..., :2].sigmoid()) * b_obj_scale)
            # pwh
            t_lbox += torch.mean(
                KDboxLoss(pi[..., 2:4].sigmoid(), t_pi[..., 2:4].sigmoid() * wh_norm_scale) * b_obj_scale)

        # Class
        if model.nc > 1:  # cls loss (only if multiple classes)
            c_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1, 1, 1, 1, model.nc)
            if kd_loss_selected == "kl":
                kl_loss = KDclsLoss(F.log_softmax(pi[..., 5:] / temperature, dim=-1),
                                    F.softmax(t_pi[..., 5:] / temperature, dim=-1)) * (temperature * temperature)
                t_lcls += torch.mean(kl_loss * c_obj_scale)
            else:
                t_lcls += torch.mean(KDclsLoss(pi[..., 5:], t_pi[..., 5:]) * c_obj_scale)

        t_lobj += torch.mean(KDobjLoss(pi[..., 4], t_pi[..., 4]) * t_obj_scale)
    t_lbox *= h['giou'] * h['kd']
    t_lobj *= h['obj'] * h['kd']
    t_lcls *= h['cls'] * h['kd']
    bs = pred[0].shape[0]  # batch size
    mkdloss = (t_lobj + t_lbox + t_lcls) * bs
    return mkdloss


def conv1x1_bn(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.SiLU()
    )


def ft(x):
    return F.normalize(x.pow(2).mean(2).mean(2).view(x.size(0), -1))


def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def ft_loss(x, y):
    # 可以修改度量函数（如余弦相似度）
    # cosin = torch.cosine_similarity(ft(x), ft(y))[0]
    return (ft(x) - ft(y)).pow(2).mean()


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()


def creat_mask(cur, labels, ratio):
    B, H, W = cur.size()
    x, y, w, h = labels[2:]
    x1 = int(((x - w / 2) * W).ceil().cpu().numpy())
    x2 = int(((x + w / 2) * W).floor().cpu().numpy())
    y1 = int(((y - h / 2) * W).ceil().cpu().numpy())
    y2 = int(((y + h / 2) * W).floor().cpu().numpy())
    cur[labels[0].cpu().numpy()][y1: y2, x1: x2] = 1 - ratio


class EFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.s_t_pair = [16, 8, 16, 32]
        # self.t_s_pair = [128, 64, 128, 256]
        self.s_t_pair = [16, 16, 32]
        self.t_s_pair = [128, 128, 256]

        self.linears = nn.ModuleList([conv1x1_bn(s, t).to("cuda:0") for s, t in zip(self.s_t_pair, self.t_s_pair)])
        # self.Ca = nn.ModuleList([CoordAtt(2 * s, 2 * s).to("cuda:0") for s in self.s_t_pair])
        # self.se1 = nn.ModuleList([SE_Block(t).to("cuda:0") for t in self.t_s_pair])
        # self.se2 = nn.ModuleList([SE_Block(s).to("cuda:0") for s in self.s_t_pair])

    def forward(self, t_f, s_f):
        device = t_f[0].fea.device
        atloss = torch.zeros(1, device=device)
        ftloss = torch.zeros(1, device=device)
        for i in range(len(t_f)):
            # t_f[i].fea = self.Ca[i](t_f[i].fea)
            # t_f[i].fea = self.se1[i](t_f[i].fea)
            # s_f[i].fea = self.se2[i](s_f[i].fea)
            atloss += at_loss(t_f[i].fea, s_f[i].fea)
            s_f[i].fea = self.linears[i](s_f[i].fea)
            ftloss += ft_loss(t_f[i].fea, s_f[i].fea)
        return atloss + ftloss, torch.cat((atloss, ftloss)).detach()


class EFKD(nn.Module):
    def __init__(self, isL=False):
        super().__init__()
        self.s_t_pair = [16, 8, 16, 24]
        self.t_s_pair = [640, 320, 640, 640]
        self.linears = nn.ModuleList([conv1x1_bn(s, t).to("cuda:0") for s, t in zip(self.s_t_pair, self.t_s_pair)])
        if isL:
            self.linears = nn.ModuleList([conv1x1_bn(s, 2 * s).to("cuda:0") for s in self.s_t_pair])

    def forward(self, targets, t_f, s_f, ratio):
        device = t_f[0].fea.device
        atloss = torch.zeros(1, device=device)
        ftloss = torch.zeros(1, device=device)
        for i in range(len(t_f)):
            b, c, h, w = s_f[i].fea.size()
            cur_mask = torch.full((b, h, w), ratio, device=device)
            for label in targets:
                creat_mask(cur_mask, label, ratio)
            atloss += wat_loss(t_f[i].fea, s_f[i].fea, cur_mask.view(b, -1))  # 计算AT Loss的第一项
            s_f[i].fea = self.linears[i](s_f[i].fea)  # 改变学生网络特征图尺度
            ftloss += ft_loss(t_f[i].fea, s_f[i].fea)  # 计算AT Loss的第二项
        return atloss + ftloss, torch.cat((atloss, ftloss)).detach()  # ftloss表示特征图损失，可以在前面加上一个权重参数


def wat_loss(x, y, mask):
    return ((at(x) - at(y)) * mask).pow(2).mean()


class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """

    def __init__(self, channels_s, channels_t, tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape

            N, C, H, W = s.shape

            # normalize in channel diemension
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau, dim=1)  # [N*C, H*W]

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (self.tau ** 2)

            losses.append(cost / (C * N))
        loss = sum(losses)

        return loss


class MGDLoss(nn.Module):
    def __init__(self, channels_s, channels_t, alpha_mgd=0.00002, lambda_mgd=0.65):
        super(MGDLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.generation = [
            nn.Sequential(
                nn.Conv2d(channel_s, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1)).to(device) for channel_s, channel in
            zip(channels_s, channels_t)
        ]

    def forward(self, y_s, y_t, layer=None):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # print(s.shape)
            # print(t.shape)
            # assert s.shape == t.shape
            if layer == "outlayer":
                idx = -1
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N

        return dis_loss


class SoftLoss:

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device
        self.softmax = nn.Softmax(dim=1)

        self.use_dfl = m.reg_max > 1
        roll_out_thr = h.min_memory if h.min_memory > 1 else 64 if h.min_memory else 0  # 64 is default

        self.assigner = TaskAlignedAssigner(topk=10,
                                            num_classes=self.nc,
                                            alpha=0.5,
                                            beta=6.0,
                                            roll_out_thr=roll_out_thr)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, teacher_preds, batch):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        # student preds
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2) \
            .split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        # teacher preds
        t_feats = teacher_preds[1] if isinstance(teacher_preds, tuple) else teacher_preds
        t_pred_distri, t_pred_scores = torch.cat([xi.view(t_feats[0].shape[0], self.no, -1) for xi in t_feats], 2) \
            .split((self.reg_max * 4, self.nc), 1)

        t_pred_scores = t_pred_scores.permute(0, 2, 1).contiguous()
        t_pred_distri = t_pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        t_pred_bboxes = self.bbox_decode(anchor_points, t_pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            t_pred_scores.detach().sigmoid(), (t_pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        # t_pred_bboxes /= stride_tensor  # teacher模型预测的boxes不需要除以stride
        # TODO: 获取教师模型预测的scores
        # t_pred_scores = self.softmax(t_pred_scores)  # 教师网络预测结果需要经过一次softmax
        # t_scores_sum = max(t_pred_scores.sum(), 1)
        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        # 教师网络预测的每一个锚框都需要计算损失
        # fg_mask = torch.ones(batch_size, pred_distri.shape[1], dtype=torch.bool).cuda()
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, t_pred_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
