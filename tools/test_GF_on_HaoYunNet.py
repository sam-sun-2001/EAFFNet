
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
from datasets.GFfloods import GFfloods
from datasets.Sen1Floods11 import Sen1Floods11
from models import unet
from models.hrnet import HighResolutionNet
from models.pureUnet import PureUnet
from models.unet import Unet
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
    test_dataset=GFfloods(root='/root/autodl-tmp/pycharm_project_666/data/GFfloodsnet/all_train_data/',#'/tmp/pycharm_project_858/data/'
                          type="test",
                          list_path='/root/autodl-tmp/pycharm_project_666/data/list/GFfloodsnet/gf_test.txt',
                          flip=False,
                          crop_size=0,
                          num_classes=2,
                          ignore_label=-1, #-1
                          base_size=512)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=6,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    focal_loss=FocalLoss(gamma=1.7,alpha=torch.tensor([0.25,0.75]),reduction="mean",ignore_index=-1)
    dice_loss=DiceLoss(ignore_index=-1)
    ssim_loss = SSIM(window_size=11, size_average=True)
    #构建Unet
    from configs import Config
    # model=HighResolutionNet(Config,in_channels=2)
    # model=PureUnet(n_channels=5,n_classes=2)
    # model=Unet(n_channels=5,n_classes=2)
    from models.deeplabv3plus import DeepLab
    model=DeepLab(num_classes=2)
    model = FullModel(model, focal_loss,dice_loss,ssim_loss)
    if model:
        print("unet有效")
    else:
        print("unet没效")
        return 0

    # 加载效果最好的pt
    model_state_file ="/root/autodl-tmp/pycharm_project_666/_73th_checkpoints/best.pt"

    # # 修改： 不加载best.pt
    # model_state_file='/tmp/pycharm_project_666/_3th_checkpoints/checkpoint.pth.tar'
    # if os.path.isfile(model_state_file):
    #     checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
    #     dct = checkpoint['state_dict']
    #     model_dict = model.state_dict()
    #     model_dict.update(dct)
    #     model.load_state_dict(model_dict)
    #     logger.info('=> loading model from {}'.format(model_state_file))
    #     model = model.cuda()
    #     logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    # 加载效果最好的pt
    # model_state_file ="/tmp/pycharm_project_666/checkpoints/best.pt"
    logger.info('=> loading model from {}'.format(model_state_file))
    pretrained_dict = torch.load(model_state_file)
    if pretrained_dict:
        print("预训练模型加载成功")
    else:
        print("没有加载best.pt")
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
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