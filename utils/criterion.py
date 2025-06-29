# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from configs import config
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch
import torch.nn as nn



class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index  # 指定忽略标签的值

    def forward(self, input, target):
        # Apply sigmoid activation to get probabilities
        input = torch.argmax(input,dim=1)
        # print(f"input:{input.shape}")?input:torch.Size([8, 256, 256])
        input_flat = input.view(input.size(0), -1)#正例概率flatten
        target_flat = target.view(target.size(0), -1)#标签
        # 忽略指定标签的像素
        if self.ignore_index is not None:
            mask = target_flat != self.ignore_index  # Shape (N, H*W)
            input_flat = input_flat[mask]  # 过滤的作用，去掉了，，，
            target_flat = target_flat[mask]  #过滤的作用，去掉了，，，
        # Compute intersection and Dice score
        intersection = (input_flat * target_flat).sum()
        dice_score = (2 * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)

        # Dice loss
        loss = 1 - dice_score

        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction='sum', ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha
        assert reduction in ['mean', 'sum'], "Reduction must be 'mean' or 'sum'."
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if target.dim() == 4:
            n, c, h, w = target.shape
            if c == 1:
                target = target.view(n, h, w)
            else:
                raise ValueError("Target tensor should have a single channel.")

        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))

        target = target.view(-1, 1)

        # 忽略指定标签的样本
        valid_mask = (target != self.ignore_index).squeeze(1)
        target = target[valid_mask]
        input = input[valid_mask]

        logpt = nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            # 将 alpha 移到与 input 相同的设备
            self.alpha = self.alpha.to(input.device)
            if self.alpha.dim() == 1 and self.alpha.size(0) == input.size(1):
                at = self.alpha.gather(0, target.data.view(-1))
                logpt = logpt * at
            else:
                raise ValueError("Alpha should be a 1D tensor with size equal to the number of classes.")

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=255, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=self.ignore_label
        )

    def forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp







class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=255, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=255,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        """
        交叉熵函数
        """
        #组合一：0.5*_ce_forward(outputs[0]，target)
        """
        score:(outputs[0]:#torch.Size([3, 19, 1024, 2048])
        target:   [3, 1024, 2048]
        """
        loss = self.criterion(score, target)
        # print(f"loss:{loss.shape}")
        #loss:torch.Size([3, 1024, 2048])
        return loss

    def _ohem_forward(self, score, target, **kwargs):
        """
        在线样本难例挖掘函数
        """
        """
        score:    outputs[1]     ([3, 19, 1024, 2048])
        target:   [3, 1024, 2048]
        """

        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)#计算每个像素的损失值 pixel_losses 是一个包含每个像素损失值的一维张量
        #标签里面 没有被忽略的位置，记为mask
        mask = target.contiguous().view(-1) != self.ignore_label
        tmp_target = target.clone()
        tmp_target[tmp_target == 255] = 0
        assert (tmp_target >= 0).all() and (tmp_target < pred.size(1)).all(), "Invalid indices in tmp_target"
        #它比较张量 tmp_target 中的每个元素是否都小于 pred 张量的第二个维度的大小
        #pred size:(3,19,1024,2048)
        #tmp_target=3,1024,2048
        #pred_gather:按照索引拿值
        #pred: 最终预测结果
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        #pred: 是经过排序后的预测概率值的张量。
        #ind: 是排序后的预测概率值在原始张量中的索引位置的张量。
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        #确定用于选择难样本的阈值
        threshold = max(min_value, self.thresh)
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        #pixel_losses.mean():3.0670037269592285
        return pixel_losses.mean()

    def forward(self, score, target):
        """
        整体的前向传播函数
        """

        # score: list [outputs[0],outputs[1]
        # outputs[0] #torch.Size([3, 19, 1024, 2048])
        # outputs[1] #torch.Size([3, 19, 1024, 2048])
        # target: labels:  [3, 1024, 2048]

        # mask = target.contiguous().view(-1) != 255
        # print("mask 中存在 False 元素")#走了这一步
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]
        balance_weights = config.LOSS.BALANCE_WEIGHTS#[0.5, 0.5] float list
        sb_weights = config.LOSS.SB_WEIGHTS#0.5
        if len(balance_weights) == len(score):
            functions = [self._ce_forward] * (len(balance_weights) - 1) + [self._ohem_forward]
            # function:      [_ce_forward,_ohem_forward]
            #组合一：0.5*_ce_forward(outputs[0])
            #组合二：0.5*_ohem_forward(outputs[1])
            #求和： 组合一+组合二


            # mask = target.contiguous().view(-1) != 255
            # if mask.all():
            #     print("mask 中所有元素都为 True")
            # else:
            #     print("mask 中存在 False 元素")#走了这一步
            return sum([
                w * func(x, target)
                for (w, x, func) in zip(balance_weights, score, functions)
            ])
        
        elif len(score) == 1:
            return sb_weights * self._ohem_forward(score[0], target)
        
        else:
            print("!!!!!gungunun")
            raise ValueError("lengths of prediction and target are not identical!")


def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0,2,3,1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True, ignore_label=None):
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM(window_size, size_average)
        self.ignore_label = ignore_label  # 设置忽略标签的值，默认为 None

    def forward(self, img1, img2, labels=None):
        # 如果提供了 labels 和 ignore_label，生成掩码
        if self.ignore_label is not None and labels is not None:
            mask = (labels != self.ignore_label).float()  # 生成掩码，忽略掉 ignore_label 区域
        else:
            mask = torch.ones_like(img1).float()  # 没有标签或没有指定 ignore_label，则全图计算

        # 使用掩码计算 SSIM，忽略掉需要忽略的标签区域
        ssim_value = self.ssim(img1 * mask, img2 * mask)

        return 1 - ssim_value  # 这里返回 1 - SSIM 作为损失

# SSIM部分保持不变
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# class SSIMLoss(nn.Module):
#     def __init__(self, window_size=11, size_average=True):
#         super(SSIMLoss, self).__init__()
#         self.ssim = SSIM(window_size, size_average)
#
#     def forward(self, img1, img2):
#         ssim_value = self.ssim(img1, img2)
#         return 1 - ssim_value  # 这里返回 1 - SSIM 作为损失

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# class SSIM(torch.nn.Module):
#     def __init__(self, window_size = 11, size_average = True):
#         super(SSIM, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = 1
#         self.window = create_window(window_size, self.channel)
#
#     def forward(self, img1, img2):
#         (_, channel, _, _) = img1.size()
#
#         if channel == self.channel and self.window.data.type() == img1.data.type():
#             window = self.window
#         else:
#             window = create_window(self.window_size, channel)
#
#             if img1.is_cuda:
#                 window = window.cuda(img1.get_device())
#             window = window.type_as(img1)
#
#             self.window = window
#             self.channel = channel
#
#
#         return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

# def ssim(img1, img2, window_size = 11, size_average = True):
#     (_, channel, _, _) = img1.size()
#     window = create_window(window_size, channel)
#
#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)
#
#     return _ssim(img1, img2, window, window_size, channel, size_average)



class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce = 20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce
        
    def forward(self, bd_pre, bd_gt):

        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        loss = bce_loss
        
        return loss
    
if __name__ == '__main__':
    a = torch.zeros(2,64,64)
    a[:,5,:] = 1
    pre = torch.randn(2,1,16,16)
    
    Loss_fc = BondaryLoss()
    loss = Loss_fc(pre, a.to(torch.uint8))

        
        
        


