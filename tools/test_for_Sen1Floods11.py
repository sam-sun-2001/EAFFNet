
import argparse
import os
import pprint

import logging
import timeit


import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from configs import config
# from configs import update_config
from datasets.Sen1Floods11 import Sen1Floods11
from models import unet
from models.hrnet import HighResolutionNet
from utils.criterion import CrossEntropy, FocalLoss, DiceLoss, SSIM
from utils.function import testval, test
from utils.utils import create_logger, FullModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    # update_config(config, args)

    return args

def main():
    args = parse_args()
    import random
    print('Seeding with', 304)
    random.seed(304)
    torch.manual_seed(304)
    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    #数据集准备
    test_dataset=Sen1Floods11(root='/root/autodl-tmp/pycharm_project_666/data/Sen1Floods11/all_train_data/',#'/tmp/pycharm_project_858/data/'
                              type="test",
                              list_path='/root/autodl-tmp/pycharm_project_666/data/list/Sen1Floods11/flood_test_data.lst',
                              flip=False,
                              crop_size=0,
                              num_classes=2,
                              ignore_label=-1, #-1
                              base_size=512)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    focal_loss=FocalLoss(gamma=1.7,alpha=torch.tensor([0.25,0.75]),reduction="mean",ignore_index=-1)
    dice_loss=DiceLoss(ignore_index=-1)
    ssim_loss = SSIM(window_size=11, size_average=True)
    #构建Unet
    from configs import Config
    # model=HighResolutionNet(Config,in_channels=2)
    # model = FullModel(model, focal_loss,dice_loss,ssim_loss)
    from models.pureUnet import PureUnet
    model=PureUnet(n_channels=2,n_classes=2)
    if model:
        print("unet有效")
    else:
        print("unet没效")
        return 0

    # 加载效果最好的pt
    model_state_file ="/root/autodl-tmp/pycharm_project_666/_v5_checkpoints/weights_88_0.35.pth"
    logger.info('=> loading model from {}'.format(model_state_file))
    checkpoint = torch.load(model_state_file)
    model.load_state_dict(checkpoint["model"])
    model = model.cuda()

    #保存预测结果的目录
    path_save_test="/root/autodl-tmp/pycharm_project_666/_16th_valid_set"
    mIoU,IoU,omission,commission,OA = testval(config,#m没错，用这个函数
                                              test_dataset,
                                              testloader,
                                              model,
                                              num_output_class=2,
                                              ignore_label=-1,
                                              sv_dir=path_save_test,sv_pred=False)

    msg = 'mIoU: {: 4.4f}, IoU: {: 4.4f}, \
                omission: {: 4.4f}, commission:{:4.4f},OA:{:4.4f}'.format(mIoU,IoU,omission,commission,OA)
    logging.info(msg)





if __name__ == '__main__':
    main()