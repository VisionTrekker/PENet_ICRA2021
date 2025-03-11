import torch
import torch.nn as nn

loss_names = ['l1', 'l2']

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0.0001).detach()      # [b, 1, 320, 1216]
        diff = target - pred                    # [b, 1, 320, 1216]
        diff = diff[valid_mask]                 # [N]   维度等于 b 张图片中有效数据个数的总和
        self.loss = (diff**2).mean()            # [1]
        return self.loss                        # 返回的是一个tensor，其存储的值为一个 batch 中所有图片 loss 的总和


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0.0001).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


# 后更
class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

        self.t_valid = 0.0001

    def forward(self, pred, gt):
        assert pred.dim() == gt.dim(), "inconsistent dimensions"

        mask = (gt > self.t_valid).type_as(pred).detach()

        d = torch.abs(pred - gt) * mask

        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask, dim=[1, 2, 3])

        loss = d / (num_valid + 1e-8)

        loss = loss.sum()

        return loss

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

        self.t_valid = 0.0001

    def forward(self, pred, gt):

        mask = (gt > self.t_valid).type_as(pred).detach()

        d = torch.pow(pred - gt, 2) * mask

        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask, dim=[1, 2, 3])

        loss = d / (num_valid + 1e-8)

        loss = loss.sum()

        return loss

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.args = args
        self.loss_dict = {}
        # self.l1Loss = L1Loss()
        # self.l2Loss = L2Loss()
        self.l1Loss = MaskedL1Loss()
        self.l2Loss = MaskedMSELoss()
        # Loss configuration : w1*l1+w2*l2+w3*l3+...
        # Ex : 1.0*L1+0.5*L2+...
        for loss_item in args.criterion.split('+'):
            weight, loss_type = loss_item.split('*')
            loss_func = loss_type + 'Loss()'

            self.loss_dict.update({loss_type: {'weight': float(weight), 'func': loss_func}})

            # loss_type = L1, L2, loss_func = L1Loss, L2Loss

    # {'L1': {'weight': 1.0, 'func': L1Loss()}, 'L2': {'weight': 1.0, 'func': L2Loss()}}

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        loss_l1w = loss_l2w = 0
        for idx, loss_type in enumerate(self.loss_dict):
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']
            if loss_func == 'l1Loss()':
                loss_l1 = self.l1Loss(pred, target)
                loss_l1w = loss['weight'] * loss_l1
            if loss_func == 'l2Loss()':
                loss_l2 = self.l2Loss(pred, target)
                loss_l2w = loss['weight'] * loss_l2

        loss_sum = loss_l1w + loss_l2w

        return loss_sum, loss_l1, loss_l2