# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
from torch import optim

from models import unet
import argparse
import pprint
import logging
import timeit
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter
from configs import Config
from utils.criterion import CrossEntropy, FocalLoss, DiceLoss, SSIM
from utils.function import train, validate
from utils.utils import create_logger, FullModel
import os
from datasets.Sen1Floods11 import Sen1Floods11
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="/tmp/pycharm_project_130/configs/cityscapes/pidnet_small_cityscapes.yaml",#配置文件的路径
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    logger, final_output_dir, tb_log_dir = create_logger(
        Config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(Config)
    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = Config.CUDNN.BENCHMARK
    cudnn.deterministic = Config.CUDNN.DETERMINISTIC
    cudnn.enabled = Config.CUDNN.ENABLED

    #batch size
    batch_size = 8
    batch_size_validate=8
    #我把第三次实验的checkpoints放在这里
    final_output_dir='/root/autodl-tmp/pycharm_project_666/_31th_checkpoints_pretrained/' #存放checkpoints以及best.pt的目录
    if not os.path.exists(final_output_dir):
        # 如果目录不存在，则创建目录
        os.makedirs(final_output_dir)
    #加载预训练权重的路径
    pretrained_dir="/root/autodl-tmp/pycharm_project_666/hrnet_pretrained/hrnetv2_w18_imagenet_pretrained.pth"
    best_mIoU = 0
    last_epoch = 0
    start = timeit.default_timer()
    end_epoch = 300
    train_dataset=Sen1Floods11(root='/root/autodl-tmp/pycharm_project_666/data/Sen1Floods11/all_train_data/',#'/tmp/pycharm_project_858/data/'
                               type="train",
                               list_path='/root/autodl-tmp/pycharm_project_666/data/list/Sen1Floods11/flood_train_data.lst',
                               flip=True,
                               crop_size=[256,256],
                               num_classes=2,
                               ignore_label=-1,
                               base_size=512)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.WORKERS,
        pin_memory=False,
        drop_last=False)
    test_dataset=Sen1Floods11(root='/root/autodl-tmp/pycharm_project_666/data/Sen1Floods11/all_train_data/',#'/tmp/pycharm_project_858/data/'
                              type="test",
                              list_path='/root/autodl-tmp/pycharm_project_666/data/list/Sen1Floods11/flood_valid_data.lst',
                              flip=False,
                              crop_size=0,
                              num_classes=2,
                              ignore_label=-1, #-1
                              base_size=512)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size_validate,
        shuffle=True,
        num_workers=Config.WORKERS,
        pin_memory=False,
        drop_last=False)
    print(f"train_dataset_len:{train_dataset.__len__()}")
    print(f"test_dataset_len:{test_dataset.__len__()}")
    # train_lr=0.01
    epoch_iters = int(train_dataset.__len__() / batch_size)
    num_iters = end_epoch * epoch_iters
    focal_loss=FocalLoss(gamma=1.7,alpha=torch.tensor([0.25,0.75]),reduction="mean",ignore_index=-1)
    dice_loss=DiceLoss(ignore_index=-1)
    ssim_loss = SSIM(window_size=11, size_average=True)
    from models.hrnet import HighResolutionNet
    model=HighResolutionNet(Config,in_channels=2)
    print("11")
    #加载预训练模型
    pretrained_dict = torch.load(pretrained_dir)
    # del pretrained_dict['conv1.weight']  # 删除不需要的键
    #加载预训练模型权重
    if pretrained_dict:
        print("预训练模型加载成功")
    else:
        print("没有加载预训练模型")
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
        # 去除 'module.' 前缀的字典
    pretrained_dict = {
        k[7:] if k.startswith('module.') else k: v
        for k, v in pretrained_dict.items()
    }
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    del model_dict['conv1.weight']
    model.load_state_dict(model_dict,strict=False)


    model = FullModel(model, focal_loss,dice_loss,ssim_loss)
    model=model.cuda()
    # optimizer
    if Config.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=Config.TRAIN.LR
        )
    else:
        raise ValueError('Only Support SGD optimizer')
    if Config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')#存放checkpoint的路径
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})#加载所有的checkpoint信息到checkpoint变量
            best_mIoU = checkpoint['best_mIoU']#获取best miou
            last_epoch = checkpoint['epoch']#获取epoch记录到哪一个epoch了
            dct = checkpoint['state_dict']#获取模型的权重！！！！！！！！
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(dct)#使用load_state_dict加载模型权重
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    if isinstance(Config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, Config.TRAIN.LR_STEP,
            Config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, Config.TRAIN.LR_STEP,
            Config.TRAIN.LR_FACTOR, last_epoch-1
        )


    #真正开始训练
    flag_rm = 1
    for epoch in range(last_epoch, end_epoch):
        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)
        train(Config, epoch, end_epoch,
              epoch_iters, Config.TRAIN.LR, num_iters,
              trainloader, optimizer, model, writer_dict)
        lr_scheduler.step()
        if flag_rm == 1 or (epoch <90 and epoch%5==0) or (epoch>90):
            mIoU,IoU,omission,commission,OA = validate(Config,
                                                       testloader, model, writer_dict,num_class=2,ignore_label=-1,train_dataset=test_dataset)
        if flag_rm == 1:
            flag_rm = 0
        logger.info('=> saving checkpoint to {}'.format(
            final_output_dir + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch+1,
            'best_mIoU': best_mIoU,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir,'checkpoint.pth.tar'))
        # if mIoU > best_mIoU:
        #     torch.save(model.state_dict(),
        #                os.path.join(final_output_dir, 'best_flood.pt'))
        if IoU > best_mIoU:
            best_mIoU = IoU
            torch.save(model.state_dict(),
                       os.path.join(final_output_dir, 'best.pt'))
        msg = 'mIoU: {:.3f}, IoU: {: 4.4f}, omission: {: 4.4f}, commission:{:4.4f},OA:{:4.4f}'.format(
            mIoU,IoU,omission,commission,OA)
        logging.info(msg)
        logging.info(best_mIoU)

    torch.save(model.state_dict(),
               os.path.join(final_output_dir, 'final_state.pt'))

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % int((end-start)/3600))
    logger.info('Done')
if __name__ == '__main__':
    main()
