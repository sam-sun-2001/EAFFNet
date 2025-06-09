import torch
import torch.nn as nn
import torch.nn.functional as F
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # 将预测值通过softmax转换为概率
        inputs = F.softmax(inputs, dim=1)

        # 将忽略索引的值设置为0
        targets = targets.clone()  # 克隆targets以避免修改原始数据
        targets[targets == self.ignore_index] = 0

        # 将预测值和目标张量重塑为 [N, C, H*W]
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # [b, c, hw]
        targets = targets.view(targets.size(0), -1)  # [b, hw]

        # 计算交集
        intersection = (inputs * targets.unsqueeze(1)).sum(dim=2)  # [b, c]

        # 计算并集
        inputs_sum = inputs.sum(dim=2)  # [b, c]
        targets_sum = targets.sum(dim=1).unsqueeze(1)  # [b, 1]
        union = inputs_sum + targets_sum

        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # 计算Dice Loss
        loss = 1 - dice.mean(dim=1).mean()  # 先对每个样本计算平均，然后对所有样本取平均

        return loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=255, reduction='mean', weight=None):
        super(FocalLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index, weight=weight)
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        # 计算交叉熵损失
        loss = self.ce_loss(logits, targets)

        # 将损失转换为Focal Loss
        pt = torch.exp(-loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * loss

        # 如果需要，应用reduction
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()

        return focal_loss