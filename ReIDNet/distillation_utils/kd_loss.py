"""
    Self Define Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # self.s_t_pair = [32, 64, 128, 256, 128, 64, 128, 256]
        # self.t_s_pair = [64, 128, 256, 512, 256, 128, 256, 512]
        self.s_t_pair = [16, 16, 32]
        self.t_s_pair = [640, 640, 1280]
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
