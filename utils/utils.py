# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config

class FullModel(nn.Module):

    #model = FullModel(model, sem_criterion, bd_criterion)

    def __init__(self, model, focal_loss,dice_loss,ssim_loss):
        super(FullModel, self).__init__()
        self.model = model
        self.focal_loss = focal_loss
        self.dice_loss = dice_loss
        self.ssim_loss=ssim_loss


    def getModel(self):
        return self.model
    def pixel_acc(self, pred, label):
        """
          :param pred: 模型预测的类别
          :param label: 标签
          :return: acc:预测对的像素/有效的总像素
        """
        # print(f"pred:{pred.shape}")
        score, preds = torch.max(pred, dim=1)#8,256,256
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def forward(self, inputs, labels,  *args, **kwargs):
        """
        @inputs 输入图像， size：
        @labels 标签，size： (N,H,W)
        @bd_gt:    size:  (N,H,W)
        """
        # print(f"正向传播：inputs:{inputs.shape}")
        outputs = self.model(inputs, *args, **kwargs)#这里的pred：256×256
        # outputs=outputs[2]
        acc  = self.pixel_acc(outputs, labels) #倒数第二个元素
        loss_focal_loss = self.focal_loss(outputs, labels.squeeze(1))
        loss_dice_loss=self.dice_loss(outputs,labels)
        ssim_loss=self.ssim_loss(torch.argmax(outputs,dim=1,keepdim=True).float(),labels.float())
        loss = loss_focal_loss + loss_dice_loss+ssim_loss
        # print(outputs[2].shape)
        return torch.unsqueeze(loss,0),outputs,  acc, [loss_focal_loss,loss_dice_loss,ssim_loss]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
                          (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)



def get_confusion_matrix(label, pred, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)#形成 (batch_size, height, width) 的数组 seg_pred
    seg_gt = np.asarray(
        label.cpu().numpy(),dtype=np.int)#将标签转换成了整型
    ignore_index = seg_gt != ignore #我们想要不是忽略的为true
    # print(f"seg_gt:{seg_gt.shape}")
    # print(f"seg_pred:{seg_pred.shape}")
    seg_gt = seg_gt[ignore_index]#过滤label
    seg_gt=seg_gt.flatten()
    seg_pred=seg_pred.flatten()
    ignore_index=ignore_index.flatten()
    seg_pred = seg_pred[ignore_index]#过滤pred
    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)#统计每种类别组合出现的次数，结果存储在 label_count 中
    confusion_matrix = np.zeros((num_class, num_class))#初始化批次的混淆矩阵
    for i_label in range(num_class): #混淆矩阵的填充
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

# def get_confusion_matrix(label, pred, num_class,size, ignore=-1):
#     """
#     Calcute the confusion matrix by given label and pred
#     """
#     # output = pred.cpu().numpy().transpose(0, 2, 3, 1)
#     print(f"pred:{pred.shape}")
#     seg_pred = np.asarray(np.argmax(pred, axis=3), dtype=np.uint8)#返回值变成 (batch_size, height, width) 的数组 seg_pred
#     print(f"seg_pred:{seg_pred.size}")#seg_pred:2097152
#     seg_gt=np.asarray(np.argmax(label.cpu().numpy(),axis=3),dtype=np.uint8)
#     # seg_gt = np.asarray(
#     #     label.cpu().numpy()[:,:size[-2],:size[-1]],dtype=np.int32)#将标签转换成了整型
#     # print(f"seg_gt:{seg_gt.shape}")#seg_gt:(8, 1, 512, 512)
#     ignore_index = seg_gt != ignore #我们想要不是忽略的为true
#     # print(f"ignore_index:{ignore_index.shape}")
#     # print(f"seg_gt:{seg_gt.shape}")
#     # print(f"seg_pred:{seg_pred.shape}")
#     seg_gt = seg_gt[ignore_index]#过滤label
#     seg_pred = seg_pred[ignore_index]#过滤pred
#     index = (seg_gt * num_class + seg_pred).astype('int32')
#     label_count = np.bincount(index)#统计每种类别组合出现的次数，结果存储在 label_count 中
#     confusion_matrix = np.zeros((num_class, num_class))#初始化批次的混淆矩阵
#     for i_label in range(num_class): #混淆矩阵的填充
#         for i_pred in range(num_class):
#             cur_index = i_label * num_class + i_pred
#             if cur_index < len(label_count):
#                 confusion_matrix[i_label,
#                                  i_pred] = label_count[cur_index]
#     return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters,
                         cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr