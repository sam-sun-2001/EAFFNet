# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
from models import unet
import argparse
import pprint
import logging
import timeit
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter
from configs import config
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
        config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(config)
    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    #batch size
    batch_size = 8
    batch_size_validate=8
    #我把第三次实验的checkpoints放在这里
    final_output_dir='/root/autodl-tmp/pycharm_project_666/_81th_checkpoints/' #存放checkpoints以及best.pt的目录
    if not os.path.exists(final_output_dir):
        # 如果目录不存在，则创建目录
        os.makedirs(final_output_dir)
    best_mIoU = 0
    last_epoch = 0
    start = timeit.default_timer()
    end_epoch = 100
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
        num_workers=config.WORKERS,
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
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=False)
    print(f"train_dataset_len:{train_dataset.__len__()}")
    print(f"test_dataset_len:{test_dataset.__len__()}")
    train_lr=0.001
    epoch_iters = int(train_dataset.__len__() / batch_size)
    num_iters = end_epoch * epoch_iters
    focal_loss=FocalLoss(gamma=1.7,alpha=torch.tensor([0.25,0.75]),reduction="mean",ignore_index=-1)
    dice_loss=DiceLoss(ignore_index=-1)
    ssim_loss = SSIM(window_size=11, size_average=True)
    #构建Unet
    # model=dsnet.DSNet(m=2,n=3,num_classes=2,planes=2,name="s128",augment=True)
    from models.unet import Unet
    model=Unet(n_channels=2,n_classes=2)
    if model:
        print("unet有效")
    else:
        print("unet没效")
        return 0
    model = FullModel(model, focal_loss,dice_loss,ssim_loss)
    model=model.cuda()
    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        params_dict = dict(model.named_parameters())
        # print(f"params_dict:{params_dict}")
        params = [{'params': list(params_dict.values()), 'lr': train_lr}]
        # print(f"params:{params}")
        optimizer = torch.optim.SGD(params,
                                    lr=train_lr,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
    else:
        raise ValueError('Only Support SGD optimizer')


    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')#存放checkpoint的路径
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})#加载所有的checkpoint信息到checkpoint变量
            best_mIoU = checkpoint['best_mIoU']#获取best miou
            last_epoch = checkpoint['epoch']#获取epoch记录到哪一个epoch了
            dct = checkpoint['state_dict']#获取模型的权重！！！！！！！！
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(dct)#使用load_state_dict加载模型权重
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))


    #真正开始训练
    flag_rm = 1
    for epoch in range(last_epoch, end_epoch):
        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)
        train(config, epoch, end_epoch,
              epoch_iters, train_lr, num_iters,
              trainloader, optimizer, model, writer_dict)
        if flag_rm == 1 or (epoch <90) or (epoch>90):
            mIoU,IoU,omission,commission,OA = validate(config,
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

# # ------------------------------------------------------------------------------
# # Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# # ------------------------------------------------------------------------------
# from models import unet
# import argparse
# import pprint
# import logging
# import timeit
# import torch
# import torch.backends.cudnn as cudnn
# import torch.optim
# from tensorboardX import SummaryWriter
# from configs import config
# from models.BASNet import BASNet
# from utils.criterion import CrossEntropy, FocalLoss, DiceLoss, SSIM
# from utils.function import train, validate
# from utils.utils import create_logger, FullModel
# import os
# from datasets.Sen1Floods11 import Sen1Floods11
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='Train segmentation network')
#     parser.add_argument('--cfg',
#                         help='experiment configure file name',
#                         default="/root/autodl-tmp/pycharm_project_130/configs/cityscapes/pidnet_small_cityscapes.yaml",#配置文件的路径
#                         type=str)
#     parser.add_argument('--seed', type=int, default=304)
#     parser.add_argument('opts',
#                         help="Modify config options using the command-line",
#                         default=None,
#                         nargs=argparse.REMAINDER)
#     args = parser.parse_args()
#     return args
#
#
#
# def main():
#     args = parse_args()
#     if args.seed > 0:
#         import random
#         print('Seeding with', args.seed)
#         random.seed(args.seed)
#         torch.manual_seed(args.seed)
#     logger, final_output_dir, tb_log_dir = create_logger(
#         config, args.cfg, 'train')
#     logger.info(pprint.pformat(args))
#     logger.info(config)
#     writer_dict = {
#         'writer': SummaryWriter(tb_log_dir),
#         'train_global_steps': 0,
#         'valid_global_steps': 0,
#     }
#
#     # cudnn related setting
#     cudnn.benchmark = config.CUDNN.BENCHMARK
#     cudnn.deterministic = config.CUDNN.DETERMINISTIC
#     cudnn.enabled = config.CUDNN.ENABLED
#
#     #batch size
#     batch_size = 8
#     batch_size_validate=8
#     #我把第三次实验的checkpoints放在这里
#     final_output_dir='/root/autodl-tmp/pycharm_project_666/_25th_checkpoints_pretrained' #存放checkpoints以及best.pt的目录
#     if not os.path.exists(final_output_dir):
#         # 如果不存在，则创建路径
#         os.makedirs(final_output_dir)
#         print(f"路径 {final_output_dir} 已创建")
#     else:
#         print(f"路径 {final_output_dir} 已存在")
#     best_mIoU = 0
#     last_epoch = 0
#     start = timeit.default_timer()
#     end_epoch = 200
#     train_dataset=Sen1Floods11(root='/root/autodl-tmp/pycharm_project_666/data/Sen1Floods11/all_train_data/',#'/root/autodl-tmp/pycharm_project_858/data/'
#                                type="train",
#                                list_path='/root/autodl-tmp/pycharm_project_666/data/list/Sen1Floods11/s1-weak-split.lst',
#                                flip=True,
#                                crop_size=[256,256],
#                                num_classes=2,
#                                ignore_label=-1,
#                                base_size=512)
#     trainloader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=config.TRAIN.SHUFFLE,
#         num_workers=config.WORKERS,
#         pin_memory=False,
#         drop_last=False)
#     test_dataset=Sen1Floods11(root='/root/autodl-tmp/pycharm_project_666/data/Sen1Floods11/all_train_data/',#'/root/autodl-tmp/pycharm_project_858/data/'
#                               type="test",
#                               list_path='/root/autodl-tmp/pycharm_project_666/data/list/Sen1Floods11/flood_valid_data.lst',
#                               flip=False,
#                               crop_size=0,
#                               num_classes=2,
#                               ignore_label=-1, #-1
#                               base_size=512)
#     testloader = torch.utils.data.DataLoader(
#         test_dataset,
#         batch_size=batch_size_validate,
#         shuffle=False,
#         num_workers=config.WORKERS,
#         pin_memory=False,
#         drop_last=False)
#     print(f"train_dataset_len:{train_dataset.__len__()}")
#     print(f"test_dataset_len:{test_dataset.__len__()}")
#     train_lr=0.01
#     epoch_iters = int(train_dataset.__len__() / batch_size)
#     num_iters = end_epoch * epoch_iters
#     focal_loss=FocalLoss(gamma=1.7,alpha=torch.tensor([0.25,0.75]),reduction="mean",ignore_index=-1)
#     dice_loss=DiceLoss(ignore_index=-1)
#     ssim_loss = SSIM(window_size=11, size_average=True)
#     #构建Unet
#     # model=dsnet.DSNet(m=2,n=3,num_classes=2,planes=2,name="s128",augment=True)
#     # model=unet.UNet(n_channels=2,n_classes=2)
#     model=BASNet(n_channels=2,n_classes=2)
#     if model:
#         print("unet有效")
#     else:
#         print("unet没效")
#         return 0
#     model = FullModel(model)
#     model=model.cuda()
#     # optimizer
#     if config.TRAIN.OPTIMIZER == 'sgd':
#         params_dict = dict(model.named_parameters())
#         # print(f"params_dict:{params_dict}")
#         params = [{'params': list(params_dict.values()), 'lr': train_lr}]
#         # print(f"params:{params}")
#         optimizer = torch.optim.SGD(params,
#                                     lr=train_lr,
#                                     momentum=config.TRAIN.MOMENTUM,
#                                     weight_decay=config.TRAIN.WD,
#                                     nesterov=config.TRAIN.NESTEROV,
#                                     )
#     else:
#         raise ValueError('Only Support SGD optimizer')
#
#
#     if config.TRAIN.RESUME:
#         model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')#存放checkpoint的路径
#         if os.path.isfile(model_state_file):
#             checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})#加载所有的checkpoint信息到checkpoint变量
#             best_mIoU = checkpoint['best_mIoU']#获取best miou
#             last_epoch = checkpoint['epoch']#获取epoch记录到哪一个epoch了
#             dct = checkpoint['state_dict']#获取模型的权重！！！！！！！！
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             model.load_state_dict(dct)#使用load_state_dict加载模型权重
#             logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
#
#
#     #真正开始训练
#     flag_rm = 1
#     for epoch in range(last_epoch, end_epoch):
#         current_trainloader = trainloader
#         if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
#             current_trainloader.sampler.set_epoch(epoch)
#         train(config, epoch, end_epoch,
#               epoch_iters, train_lr, num_iters,
#               trainloader, optimizer, model, writer_dict)
#         if flag_rm == 1 or (epoch < 70 and epoch%5==0) or (epoch>=70):
#             mIoU,IoU,omission,commission,OA = validate(config,
#                                                        testloader, model, writer_dict,num_class=2,ignore_label=-1,train_dataset=test_dataset)
#         if flag_rm == 1:
#             flag_rm = 0
#         logger.info('=> saving checkpoint to {}'.format(
#             final_output_dir + 'checkpoint.pth.tar'))
#         torch.save({
#             'epoch': epoch+1,
#             'best_mIoU': best_mIoU,
#             'state_dict': model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#         }, os.path.join(final_output_dir,'checkpoint.pth.tar'))
#         # if mIoU > best_mIoU:
#         #     torch.save(model.state_dict(),
#         #                os.path.join(final_output_dir, 'best_flood.pt'))
#         if IoU > best_mIoU:
#             best_mIoU = IoU
#             torch.save(model.state_dict(),
#                        os.path.join(final_output_dir, 'best.pt'))
#         msg = 'mIoU: {:.3f}, IoU: {: 4.4f}, omission: {: 4.4f}, commission:{:4.4f},OA:{:4.4f},bestIoU: {: 4.4f}'.format(
#             mIoU,IoU,omission,commission,OA,best_mIoU)
#         logging.info(msg)
#
#     torch.save(model.state_dict(),
#                os.path.join(final_output_dir, 'final_state.pt'))
#
#     writer_dict['writer'].close()
#     end = timeit.default_timer()
#     logger.info('Hours: %d' % int((end-start)/3600))
#     logger.info('Done')
#
# if __name__ == '__main__':
#     main()
