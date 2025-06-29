# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import logging
import os
import time
import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate



def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()
    print("开始训练")
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_dice_loss = AverageMeter()
    avg_SSIM_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    # Trainloader来加载数据
    for i_iter, batch in enumerate(trainloader, 0):
        images, labels = batch
        images = images.float().cuda()
        labels = labels.long().cuda()
        #前向传播
        losses, score,acc, loss_list = model(images, labels)
        loss = losses.mean()
        acc  = acc.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()
        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        avg_sem_loss.update(loss_list[0].mean().item())
        avg_dice_loss.update(loss_list[1].mean().item())
        avg_SSIM_loss.update(loss_list[1].mean().item())
        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f},dice loss: {:.6f},ssim loss: {:.6f}' .format(
                epoch, num_epoch, i_iter, epoch_iters,
                batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                ave_acc.average(), avg_sem_loss.average(),avg_dice_loss.average(),avg_SSIM_loss.average())
            logging.info(msg)
    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1
def train_HSEA(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()
    print("开始训练")
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_dice_loss = AverageMeter()
    avg_SSIM_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    # Trainloader来加载数据
    for i_iter, batch in enumerate(trainloader, 0):
        images, labels = batch
        images = images.float().cuda()
        labels = labels.long().cuda()
        #前向传播
        losses, score,acc, loss_list = model(images, labels)
        loss = losses.mean()
        acc  = acc.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()
        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        avg_sem_loss.update(loss_list[0].mean().item())
        avg_dice_loss.update(loss_list[1].mean().item())
        avg_SSIM_loss.update(loss_list[1].mean().item())
        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f},dice loss: {:.6f},ssim loss: {:.6f}' .format(
                epoch, num_epoch, i_iter, epoch_iters,
                batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                ave_acc.average(), avg_sem_loss.average(),avg_dice_loss.average(),avg_SSIM_loss.average())
            logging.info(msg)
    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1
def validate(config, testloader, model, writer_dict,num_class,ignore_label,train_dataset):
    print("验证函数开始")
    model.eval()
    ave_loss = AverageMeter()
    mIou_=[]
    tpi_sum=[]
    tni_sum=[]
    fpi_sum=[]
    fni_sum=[]
    accuracy=[]
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, size,labels,name = batch
            size = labels.size()
            # print(f"size:{size}")#size:torch.Size([8, 1, 512, 512])
            image = image.float().cuda()
            if torch.isnan(image).any():
                print("张量中有 NaN 值")
            # 检查 Inf
            if torch.isinf(image).any():
                print("张量中有 Inf 值")
            labels = labels.long().cuda()
            losses, preds, acc, _ = model(image, labels)
            if torch.isnan(preds).any():
                print("张量中有 NaN 值")
            # 检查 Inf
            if torch.isinf(preds).any():
                print("张量中有 Inf 值")
            b,c,h,w=preds.shape
            preds = preds.cpu().numpy()#转换成数组，方便循环
            preds=np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)#变成了三维
            preds = np.reshape(preds, (b,1,h,w))
            label=labels.cpu().numpy()#仍然是思维
            assert preds.shape==label.shape
            for i in range(b):
                label_i = label[i, 0, :, :]#二维
                if(np.all(label_i==ignore_label)):
                    continue
                if(np.all(label_i==0)):#只计算含有洪水的区域
                    continue
                pred_i = preds[i, 0, :, :]
                filter = label_i != ignore_label
                label_i = label_i[filter]
                pred_i = pred_i[filter]
                #计算洪水这个类别的iou
                tpi = np.sum((label_i == 1) & (pred_i == 1))
                tpi_sum.append(tpi)
                fni = np.sum((label_i == 0) & (pred_i == 1))
                fni_sum.append(fni)
                fpi = np.sum((label_i == 1) & (pred_i == 0))
                fpi_sum.append(fpi)
                tni = np.sum((label_i == 0) & (pred_i == 0))
                tni_sum.append(tni)
                if (tpi+fpi+tni+fni )==0:
                    print(name[i])
                accuracyi = (tpi+tni)/(tpi+fpi+tni+fni)
                accuracy.append(accuracyi)
                Ioui=tpi/(fni+fpi+tpi)
                mIou_.append(Ioui)
            if idx % 5 == 0:
                print(idx)
            loss = losses.mean()
            ave_loss.update(loss.item())
    IoU=np.mean(mIou_)
    mIoU=np.sum(tpi_sum)/((np.sum(fni_sum)+np.sum(fpi_sum)+np.sum(tpi_sum)))
    omission=np.sum(fni_sum)/(np.sum(fni_sum)+np.sum(tpi_sum))#漏保
    commission=np.sum(fpi_sum)/(np.sum(fpi_sum)+np.sum(tni_sum))#误报
    OA=np.mean(accuracy)
    print("验证函数结束")
    return mIoU,IoU,omission,commission,OA

def validate_HSEI(config, testloader, model, writer_dict,num_class,ignore_label,train_dataset):
    print("验证函数开始")
    model.eval()
    ave_loss = AverageMeter()
    mIou_=[]
    tpi_sum=[]
    tni_sum=[]
    fpi_sum=[]
    fni_sum=[]
    accuracy=[]
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, labels= batch
            size = labels.size()
            # print(f"size:{size}")#size:torch.Size([8, 1, 512, 512])
            image = image.float().cuda()
            if torch.isnan(image).any():
                print("张量中有 NaN 值")
            # 检查 Inf
            if torch.isinf(image).any():
                print("张量中有 Inf 值")
            labels = labels.long().cuda()
            losses, preds, acc, _ = model(image, labels)
            if torch.isnan(preds).any():
                print("张量中有 NaN 值")
            # 检查 Inf
            if torch.isinf(preds).any():
                print("张量中有 Inf 值")
            b,c,h,w=preds.shape
            preds = preds.cpu().numpy()#转换成数组，方便循环
            preds=np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)#变成了三维
            preds = np.reshape(preds, (b,1,h,w))
            label=labels.cpu().numpy()#仍然是思维
            assert preds.shape==label.shape
            for i in range(b):
                label_i = label[i, 0, :, :]#二维
                if(np.all(label_i==ignore_label)):
                    continue
                if(np.all(label_i==0)):#只计算含有洪水的区域
                    continue
                pred_i = preds[i, 0, :, :]
                filter = label_i != ignore_label
                label_i = label_i[filter]
                pred_i = pred_i[filter]
                #计算洪水这个类别的iou
                tpi = np.sum((label_i == 1) & (pred_i == 1))
                tpi_sum.append(tpi)
                fni = np.sum((label_i == 0) & (pred_i == 1))
                fni_sum.append(fni)
                fpi = np.sum((label_i == 1) & (pred_i == 0))
                fpi_sum.append(fpi)
                tni = np.sum((label_i == 0) & (pred_i == 0))
                tni_sum.append(tni)
                # if (tpi+fpi+tni+fni )==0:
                #     print(name[i])
                accuracyi = (tpi+tni)/(tpi+fpi+tni+fni)
                accuracy.append(accuracyi)
                Ioui=tpi/(fni+fpi+tpi)
                mIou_.append(Ioui)
            if idx % 5 == 0:
                print(idx)
            loss = losses.mean()
            ave_loss.update(loss.item())
    IoU=np.mean(mIou_)
    mIoU=np.sum(tpi_sum)/((np.sum(fni_sum)+np.sum(fpi_sum)+np.sum(tpi_sum)))
    omission=np.sum(fni_sum)/(np.sum(fni_sum)+np.sum(tpi_sum))#漏保
    commission=np.sum(fpi_sum)/(np.sum(fpi_sum)+np.sum(tni_sum))#误报
    OA=np.mean(accuracy)
    print("验证函数结束")
    return mIoU,IoU,omission,commission,OA
# def testval(config, test_dataset, testloader, model,num_output_class,ignore_label,
#             sv_dir='./', sv_pred=False):
#     model.eval()#模型开始进行评估
#     confusion_matrix = np.zeros((num_output_class, num_output_class))#初始化一个2×2的混淆矩阵
#     with torch.no_grad():#不进行梯度下降
#         for index, batch in enumerate(testloader): #获取一个批次的照片
#             image, size,label,name = batch
#             size=label.size()
#             image=image.float().cuda()#起到一个数据格式转换的作用,将numpy数组转换成tensor才能放在gpu上
#             label=label.long().cuda()# int16-》int64
#             pred = test_dataset.single_scale_inference(config, model, image)#算出这个批次的得分（6,2,512,512）
#             confusion_matrix += get_confusion_matrix(#得到这个批次的混淆矩阵后，加到全局的混淆矩阵中
#                 label,
#                 pred,
#                 num_class=num_output_class,
#                 ignore=ignore_label)
#             #对测试图像进行可视化，进行保存
#             if sv_pred:
#                 sv_path = os.path.join(sv_dir, 'test_results')
#                 if not os.path.exists(sv_path):
#                     os.mkdir(sv_path)
#                 # print(f"pred:{pred.shape}")#pred:torch.Size([6, 2, 512, 512])
#                 test_dataset.save_pred(pred, sv_path,name)
#             if index % 5 == 0:# 这一段就是方便阅读，没什么实际意义，可以不用看
#                 logging.info('processing: %d images' % index)
#     pos = confusion_matrix.sum(1)#对每一行进行求和
#     res = confusion_matrix.sum(0)#对每一列进行求和
#     tp = np.diag(confusion_matrix)#提取对角线元素：每个类别预测对的
#     pixel_acc = tp.sum()/pos.sum() #总的预测正确率
#     mean_acc = (tp/np.maximum(1.0, pos)).mean()#即模型在每个类别上的准确率的平均值
#     IoU_array = (tp / np.maximum(1.0, pos + res - tp))#在某个类别上预测区域与实际区域的重叠程度
#     mean_IoU = IoU_array.mean()
#     TP=confusion_matrix[0][0]                      #真正的正例。 标签：正，预测：正
#     FP=confusion_matrix[1][0]                      #假的正例    标签：负  预测：正
#     TN=confusion_matrix[1][1]
#     FN=confusion_matrix[0][1]
#     precision = TP / (TP + FP) if (TP + FP) != 0 else 0# 计算 Precision，添加防止除零的检查
#     recall = TP / (TP + FN) if (TP + FN) != 0 else 0# 计算 Recall，修正公式并添加除零检查
#     F1=(2*precision*recall)/(precision+recall) if (precision+recall) != 0 else 0#F1
#     precision = round(precision, 4)#high!高精确率表示模型在预测洪水时有较少的误报。 # 输出精确到小数点后四位
#     recall = round(recall, 4)#high!高准确率代表模型在预测洪水时要有漏报率低！
#     return mean_IoU, IoU_array, pixel_acc, mean_acc,precision,recall,F1
def test_for_paper(config, test_dataset, testloader, model,num_output_class,ignore_label,
            sv_dir='./', sv_pred=False):
    model.eval()#模型开始进行评估
    mIou_=[]
    tpi_sum=[]
    tni_sum=[]
    fpi_sum=[]
    fni_sum=[]
    accuracy=[]
    precision=[]
    recall=[]
    F1=[]
    with torch.no_grad():#不进行梯度下降
        for index, batch in enumerate(testloader): #获取一个批次的照片
            image, size,label,name = batch
            if torch.isnan(image).any():
                print("张量中有 NaN 值")
                # 检查 Inf
            if torch.isinf(image).any():
                print("张量中有 Inf 值")
            image=image.float().cuda()#起到一个数据格式转换的作用,将numpy数组转换成tensor才能放在gpu上
            label=label.long().cuda()# int16-》int64
            pred = test_dataset.single_scale_inference(config, model, image)#算出这个批次的得分（6,2,512,512）
            b,c,h,w=label.shape
            pred = pred.cpu().numpy()#转换成数组，方便循环
            pred=np.asarray(np.argmax(pred, axis=1), dtype=np.uint8)#变成了三维
            pred = np.reshape(pred, (b,1,h,w))
            label=label.cpu().numpy()#仍然是思维
            assert pred.shape==label.shape
            for i in range(b):
                label_i = label[i, 0, :, :]
                if(np.all(label_i==ignore_label)):
                    continue
                if(np.all(label_i==0)):#只计算含有洪水的区域
                    continue
                pred_i = pred[i, 0, :, :]
                filter = label_i != ignore_label
                label_i = label_i[filter]
                pred_i = pred_i[filter]
                #计算洪水这个类别的iou
                tpi = np.sum((label_i == 1) & (pred_i == 1))
                fni = np.sum((label_i == 0) & (pred_i == 1))
                fpi = np.sum((label_i == 1) & (pred_i == 0))
                tni = np.sum((label_i == 0) & (pred_i == 0))
                tni_sum.append(tni)
                fpi_sum.append(fpi)
                fni_sum.append(fni)
                tpi_sum.append(tpi)
                Ioui=tpi/(fni+fpi+tpi)
                mIou_.append(Ioui)
                if (tpi+fpi==0):
                    preci=0
                else:
                    preci=tpi/(tpi+fpi)
                precision.append(preci)
                if(tpi+fni==0):
                    recal=0
                else:
                    recal=(tpi+0.000001)/(tpi+fni++0.000001)
                recall.append(recal)
                if (tpi+fpi+tni+fni==0):
                    accuracyi=0
                else:
                    accuracyi = (tpi+tni)/(tpi+fpi+tni+fni)
                accuracy.append(accuracyi)
                if(preci+recal==0):
                    f1=0
                else:
                    f1=(2*preci*recal+0.000001)/(preci+recal+0.000001)
                F1.append(f1)
            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path,name)
            if index % 5 == 0:# 这一段就是方便阅读，没什么实际意义，可以不用看
                logging.info('processing: %d images' % index)
    IoU=np.mean(mIou_)
    OA=np.mean(accuracy)
    precision_=np.mean(precision)
    recall_=np.mean(recall)
    f1_=np.mean(F1)

    return OA,IoU,precision_,recall_,f1_

def testval(config, test_dataset, testloader, model,num_output_class,ignore_label,
            sv_dir='./', sv_pred=False):
    model.eval()#模型开始进行评估
    mIou_=[]
    tpi_sum=[]
    tni_sum=[]
    fpi_sum=[]
    fni_sum=[]
    accuracy=[]
    with torch.no_grad():#不进行梯度下降
        for index, batch in enumerate(testloader): #获取一个批次的照片
            image, size,label,name = batch
            if torch.isnan(image).any():
                print("张量中有 NaN 值")
                # 检查 Inf
            if torch.isinf(image).any():
                print("张量中有 Inf 值")
            image=image.float().cuda()#起到一个数据格式转换的作用,将numpy数组转换成tensor才能放在gpu上
            label=label.long().cuda()# int16-》int64
            pred = test_dataset.single_scale_inference(config, model, image)#算出这个批次的得分（6,2,512,512）
            b,c,h,w=label.shape
            pred = pred.cpu().numpy()#转换成数组，方便循环
            pred=np.asarray(np.argmax(pred, axis=1), dtype=np.uint8)#变成了三维
            pred = np.reshape(pred, (b,1,h,w))
            label=label.cpu().numpy()#仍然是思维
            assert pred.shape==label.shape
            for i in range(b):
                label_i = label[i, 0, :, :]
                if(np.all(label_i==ignore_label)):
                    continue
                if(np.all(label_i==0)):#只计算含有洪水的区域
                    continue
                pred_i = pred[i, 0, :, :]
                filter = label_i != ignore_label
                label_i = label_i[filter]
                pred_i = pred_i[filter]
                #计算洪水这个类别的iou
                tpi = np.sum((label_i == 1) & (pred_i == 1))
                fni = np.sum((label_i == 0) & (pred_i == 1))
                fpi = np.sum((label_i == 1) & (pred_i == 0))
                tni = np.sum((label_i == 0) & (pred_i == 0))
                tni_sum.append(tni)
                fpi_sum.append(fpi)
                fni_sum.append(fni)
                tpi_sum.append(tpi)
                Ioui=tpi/(fni+fpi+tpi)
                mIou_.append(Ioui)
                accuracyi = (tpi+tni)/(tpi+fpi+tni+fni)
                accuracy.append(accuracyi)
            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path,name)
            if index % 5 == 0:# 这一段就是方便阅读，没什么实际意义，可以不用看
                logging.info('processing: %d images' % index)
    IoU=np.mean(mIou_)
    mIoU=np.sum(tpi_sum)/(np.sum(fni_sum)+np.sum(fpi_sum)+np.sum(tpi_sum))
    omission=np.sum(fni_sum)/(np.sum(fni_sum)+np.sum(tpi_sum))#漏保
    commission=np.sum(fpi_sum)/(np.sum(fpi_sum)+np.sum(tni_sum))#误报
    OA=np.mean(accuracy)
    return mIoU,IoU,omission,commission,OA

def test(config, test_dataset, testloader, model,
         sv_dir='./', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, size,label = batch
            pred = test_dataset.single_scale_inference(
                config,
                model,
                image.cuda())

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path)


def testHIEA(config, test_dataset, testloader, model,num_output_class,ignore_label,
            sv_dir='./', sv_pred=False):
    model.eval()#模型开始进行评估
    mIou_=[]
    tpi_sum=[]
    tni_sum=[]
    fpi_sum=[]
    fni_sum=[]
    accuracy=[]
    with torch.no_grad():#不进行梯度下降
        for index, batch in enumerate(testloader): #获取一个批次的照片
            image, label,name = batch
            if torch.isnan(image).any():
                print("张量中有 NaN 值")
                # 检查 Inf
            if torch.isinf(image).any():
                print("张量中有 Inf 值")
            image=image.float().cuda()#起到一个数据格式转换的作用,将numpy数组转换成tensor才能放在gpu上
            label=label.long().cuda()# int16-》int64
            pred = test_dataset.single_scale_inference(config, model, image)#算出这个批次的得分（6,2,512,512）
            # 获取唯一值
            b,c,h,w=label.shape
            pred = pred.cpu().numpy()#转换成数组，方便循环
            print(pred.shape)#(1, 2, 256, 256) socre
            pred=np.asarray(np.argmax(pred, axis=1), dtype=np.uint8)#变成了三维
            unique_values = np.unique(pred)
            # 打印唯一值
            print("Unique values:", unique_values)
            print(pred.shape)#(1, 256, 256)
            # label=label.cpu().numpy()#仍然是思维
            print(label.shape)
            prediction=pred
            pred=pred.reshape(1,1,256,256)
            assert pred.shape==label.shape
            for i in range(b):
                label_i = label[i, 0, :, :].cpu().numpy()
                if(np.all(label_i==ignore_label)):
                    continue
                if(np.all(label_i==0)):#只计算含有洪水的区域
                    continue
                pred_i = pred[i, 0, :, :]
                filter = label_i != ignore_label
                label_i = label_i[filter]
                pred_i = pred_i[filter]
                #计算洪水这个类别的iou
                tpi = np.sum((label_i == 1) & (pred_i == 1))
                fni = np.sum((label_i == 0) & (pred_i == 1))
                fpi = np.sum((label_i == 1) & (pred_i == 0))
                tni = np.sum((label_i == 0) & (pred_i == 0))
                tni_sum.append(tni)
                fpi_sum.append(fpi)
                fni_sum.append(fni)
                tpi_sum.append(tpi)
                Ioui=tpi/(fni+fpi+tpi)
                mIou_.append(Ioui)
                accuracyi = (tpi+tni)/(tpi+fpi+tni+fni)
                accuracy.append(accuracyi)
            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(prediction, sv_path,name)
            if index % 5 == 0:# 这一段就是方便阅读，没什么实际意义，可以不用看
                logging.info('processing: %d images' % index)
    IoU=np.mean(mIou_)
    mIoU=np.sum(tpi_sum)/(np.sum(fni_sum)+np.sum(fpi_sum)+np.sum(tpi_sum))
    omission=np.sum(fni_sum)/(np.sum(fni_sum)+np.sum(tpi_sum))#漏保
    commission=np.sum(fpi_sum)/(np.sum(fpi_sum)+np.sum(tni_sum))#误报
    OA=np.mean(accuracy)
    return mIoU,IoU,omission,commission,OA
# # ------------------------------------------------------------------------------
# # Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# # ------------------------------------------------------------------------------
#
# import logging
# import os
# import time
# import cv2
# import numpy as np
# from tqdm import tqdm
#
# import torch
# from torch.nn import functional as F
#
# from utils.utils import AverageMeter
# from utils.utils import get_confusion_matrix
# from utils.utils import adjust_learning_rate
#
#
#
# def train(config, epoch, num_epoch, epoch_iters, base_lr,
#           num_iters, trainloader, optimizer, model, writer_dict):
#     # Training
#     model.train()
#     print("开始训练")
#     ave_loss = AverageMeter()
#     ave_loss0=AverageMeter()
#     cur_iters = epoch*epoch_iters
#     # Trainloader来加载数据
#     for i_iter, batch in enumerate(trainloader, 0):
#         images, labels = batch
#         images = images.float().cuda()
#         labels = labels.float().cuda()
#         # print(labels.shape) 是四维的！
#         #前向传播
#         losses,loss0, score = model(images, labels)
#         loss=losses.mean()
#         loss0=loss0.mean()
#         ave_loss.update(loss.item())
#         ave_loss0.update(loss0.item())
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         lr = adjust_learning_rate(optimizer,
#                                   base_lr,
#                                   num_iters,
#                                   i_iter+cur_iters)
#
#         if i_iter % config.PRINT_FREQ == 0:
#             msg = 'Epoch: [{}/{}] Iter:[{}/{}],  ' \
#                   'lr: {}, Loss: {:.6f},Loss0: {:.6f}' .format(
#                       epoch, num_epoch, i_iter, epoch_iters,
#                       [x['lr'] for x in optimizer.param_groups],ave_loss.average(),ave_loss0.average())
#             logging.info(msg)
#
#
# # def validate(config, testloader, model, writer_dict,num_class,ignore_label,train_dataset):
# #     model.eval()
# #     ave_loss = AverageMeter()
# #     confusion_matrix = np.zeros(
# #         ( num_class, num_class)) #(2,2)
# #     with torch.no_grad():
# #         for idx, batch in enumerate(testloader):
# #             image, size,labels,name = batch
# #             size = labels.size()
# #             # print(f"size:{size}")#size:torch.Size([8, 1, 512, 512])
# #             image = image.float().cuda()
# #             labels = labels.long().cuda()
# #             # print(label.shape)
# #             losses, preds, acc, _ = model(image, labels)
# #             # #小改动的开始
# #             # sv_path = '/tmp/pycharm_project_666/validation_veiw_image'
# #             # if not os.path.exists(sv_path):
# #             #     os.mkdir(sv_path)
# #             #     # print(f"pred:{pred.shape}")#pred:torch.Size([6, 2, 512, 512])
# #             # train_dataset.save_pred(pred, sv_path,name)
# #             # print("图像已经保存")
# #             #小改动的结束
# #             if not isinstance(preds, (list, tuple)):
# #                 preds = [preds]
# #                 # print("shabiba ni ") #每次都会运行
# #             for i, x in enumerate(preds):
# #                 # print(f"x:{x.shape}")#x:torch.Size([8, 2, 512, 512])
# #                 x = F.interpolate(
# #                     input=x, size=size[-2:],
# #                     mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
# #                 )
# #                 print(f"i:{i}")
# #                 confusion_matrix += get_confusion_matrix(
# #                     labels,
# #                     x,
# #                     num_class,
# #                     ignore_label
# #                 )
# #             if idx % 5 == 0:
# #                 print(idx)
# #             loss = losses.mean()
# #             ave_loss.update(loss.item())
# #     pos = confusion_matrix.sum(1)   # 每个实际类别的计数
# #     res = confusion_matrix.sum(0)   # 每个预测类别的计数
# #     tp = np.diag(confusion_matrix)  # 取主对角线
# #     IoU_array = tp / np.maximum(1.0, pos + res - tp)  # IoU 计算
# #     mean_IoU = IoU_array.mean()
# #     TP=confusion_matrix[0][0]                      #真正的正例。 标签：正，预测：正
# #     FP=confusion_matrix[1][0]                      #假的正例    标签：负  预测：正
# #     TN=confusion_matrix[1][1]
# #     FN=confusion_matrix[0][1]
# #     precision = TP / (TP + FP) if (TP + FP) != 0 else 0# 计算 Precision，添加防止除零的检查
# #     recall = TP / (TP + FN) if (TP + FN) != 0 else 0# 计算 Recall，修正公式并添加除零检查
# #     F1=(2*precision*recall)/(precision+recall) if (precision+recall) != 0 else 0#F1
# #     precision = round(precision, 4)#high!高精确率表示模型在预测洪水时有较少的误报。# 输出精确到小数点后四位
# #     recall = round(recall, 4)#high!高准确率代表模型在预测洪水时要有漏报率低！
# #     logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
# #     writer = writer_dict['writer']
# #     global_steps = writer_dict['valid_global_steps']
# #     writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
# #     writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
# #     writer_dict['valid_global_steps'] = global_steps + 1
# #     return ave_loss.average(), mean_IoU, IoU_array,precision,recall,F1
#
#
# # def testval(config, test_dataset, testloader, model,num_output_class,ignore_label,
# #             sv_dir='./', sv_pred=False):
# #     model.eval()#模型开始进行评估
# #     iou_flood=[]
# #     iou_bg=[]
# #     recall_flood=[]
# #     precision_flood=[]
# #     F1_flood=[]
# #     with torch.no_grad():#不进行梯度下降
# #         for index, batch in enumerate(testloader): #获取一个批次的照片
# #             image, size,label,name = batch
# #             size=label.size()
# #             image=image.float().cuda()#起到一个数据格式转换的作用,将numpy数组转换成tensor才能放在gpu上
# #             label=label.long().cuda()# int16-》int64
# #             pred = test_dataset.single_scale_inference(config, model, image)#算出这个批次的得分（6,2,512,512）
# #             b,c,h,w=pred.shape
# #             pred = pred.cpu().numpy()#转换成数组，方便循环
# #             pred=np.asarray(np.argmax(pred, axis=1), dtype=np.uint8)#变成了三维
# #             pred = np.reshape(pred, (b,1,h,w))
# #             label=label.cpu().numpy()#仍然是思维
# #             assert pred.shape==label.shape
# #             for i in range(b):
# #                 label_=label[i,:,:,:]
# #                 pred_=label[i,:,:,:]
# #                 filter=pred_!=ignore_label
# #                 label_=label_[filter]
# #                 pred_=pred_[filter]
# #                 #计算洪水这个类别的iou
# #                 #计算预测洪水并且标签也是洪水的，也就是交集
# #                 num_positive = np.sum((label_ == 1) & (pred_ == 1))
# #                 num_union = np.sum((label_ == 1) | (pred_ == 1))  # 并集：标签或预测为洪水的区域
# #                 iou_for_flood=num_positive/num_union
# #                 iou_flood.append(iou_for_flood)
# #                 #背景iou
# #                 num_positive2 = np.sum((label_ == 0) & (pred_ == 0))
# #                 num_union2 = np.sum((label_ == 0) | (pred_ == 0))  # 并集：标签或预测为洪水的区域
# #                 iou_for_bg=num_positive2/num_union2
# #                 iou_bg.append(iou_for_bg)
# #                 #正例的precision
# #                 tp = np.sum((label_ == 1) & (pred_ == 1))
# #                 fp = np.sum((label_ == 0) & (pred_ == 1))
# #                 if tp + fp == 0:
# #                     precision = float('nan')
# #                 else:
# #                     precision = tp / (tp + fp)
# #                 precision_flood.append(precision)
# #                 #正例的recl
# #                 # 计算 TP 和 FN
# #                 tp2 = np.sum((label_ == 1) & (pred_ == 1))  # 真正例，标签和预测均为洪水
# #                 fn2 = np.sum((label_ == 1) & (pred_ == 0))  # 假负例，标签为洪水但预测为非洪水
# #                 if tp2 + fn2 == 0:
# #                     recall = float('nan')  # 如果没有实际为洪水的像素，Recall 设置为 NaN
# #                 else:
# #                     recall = tp2 / (tp2 + fn2)  # 召回率公式：TP / (TP + FN)zzz
# #                 print(f"Recall for flood (positive class): {recall}")
# #
# #
# #             #正例的recall
# #
# #             #正例的F1
# #
# #             #对测试图像进行可视化，进行保存
# #             if sv_pred:
# #                 sv_path = os.path.join(sv_dir, 'test_results')
# #                 if not os.path.exists(sv_path):
# #                     os.mkdir(sv_path)
# #                 # print(f"pred:{pred.shape}")#pred:torch.Size([6, 2, 512, 512])
# #                 test_dataset.save_pred(pred, sv_path,name)
# #             if index % 5 == 0:# 这一段就是方便阅读，没什么实际意义，可以不用看
# #                 logging.info('processing: %d images' % index)
# #     print(iou_flood)
# #     return iou_flood, [iou_bg,iou_flood], pixel_acc, mean_acc,precision,recall,F1
# # def testval(config, test_dataset, testloader, model,num_output_class,ignore_label,
# #             sv_dir='./', sv_pred=False):
# #     model.eval()#模型开始进行评估
# #     confusion_matrix = np.zeros((num_output_class, num_output_class))#初始化一个2×2的混淆矩阵
# #     with torch.no_grad():#不进行梯度下降
# #         for index, batch in enumerate(testloader): #获取一个批次的照片
# #             image, size,label,name = batch
# #             size=label.size()
# #             image=image.float().cuda()#起到一个数据格式转换的作用,将numpy数组转换成tensor才能放在gpu上
# #             label=label.long().cuda()# int16-》int64
# #             pred,_,_,_ = test_dataset.single_scale_inference(config, model, image)#算出这个批次的得分（6,2,512,512）
# #             confusion_matrix += get_confusion_matrix(#得到这个批次的混淆矩阵后，加到全局的混淆矩阵中
# #                 label,
# #                 pred,
# #                 num_class=num_output_class,
# #                 ignore=ignore_label)
# #             #对测试图像进行可视化，进行保存
# #             if sv_pred:
# #                 sv_path = os.path.join(sv_dir, 'test_results')
# #                 if not os.path.exists(sv_path):
# #                     os.mkdir(sv_path)
# #                 # print(f"pred:{pred.shape}")#pred:torch.Size([6, 2, 512, 512])
# #                 test_dataset.save_pred(pred, sv_path,name)
# #             if index % 5 == 0:# 这一段就是方便阅读，没什么实际意义，可以不用看
# #                 logging.info('processing: %d images' % index)
# #     pos = confusion_matrix.sum(1)#对每一行进行求和
# #     res = confusion_matrix.sum(0)#对每一列进行求和
# #     tp = np.diag(confusion_matrix)#提取对角线元素：每个类别预测对的
# #     pixel_acc = tp.sum()/pos.sum() #总的预测正确率
# #     mean_acc = (tp/np.maximum(1.0, pos)).mean()#即模型在每个类别上的准确率的平均值
# #     IoU_array = (tp / np.maximum(1.0, pos + res - tp))#在某个类别上预测区域与实际区域的重叠程度
# #     mean_IoU = IoU_array.mean()
# #     TP=confusion_matrix[0][0]                      #真正的正例。 标签：正，预测：正
# #     FP=confusion_matrix[1][0]                      #假的正例    标签：负  预测：正
# #     TN=confusion_matrix[1][1]
# #     FN=confusion_matrix[0][1]
# #     precision = TP / (TP + FP) if (TP + FP) != 0 else 0# 计算 Precision，添加防止除零的检查
# #     recall = TP / (TP + FN) if (TP + FN) != 0 else 0# 计算 Recall，修正公式并添加除零检查
# #     F1=(2*precision*recall)/(precision+recall) if (precision+recall) != 0 else 0#F1
# #     precision = round(precision, 4)#high!高精确率表示模型在预测洪水时有较少的误报。 # 输出精确到小数点后四位
# #     recall = round(recall, 4)#high!高准确率代表模型在预测洪水时要有漏报率低！
# #     return mean_IoU, IoU_array, pixel_acc, mean_acc,precision,recall,F1
# def testval(config, test_dataset, testloader, model,num_output_class,ignore_label,
#             sv_dir='./', sv_pred=False):
#     model.eval()#模型开始进行评估
#     mIou_=[]
#     tpi_sum=[]
#     tni_sum=[]
#     fpi_sum=[]
#     fni_sum=[]
#     accuracy=[]
#     with torch.no_grad():#不进行梯度下降
#         for index, batch in enumerate(testloader): #获取一个批次的照片
#             image, size,label,name = batch
#             image=image.float().cuda()#起到一个数据格式转换的作用,将numpy数组转换成tensor才能放在gpu上
#             label=label.long().cuda()# int16-》int64
#             pred,_,_,_ = test_dataset.single_scale_inference(config, model, image)#算出这个批次的得分（6,2,512,512）
#             b,c,h,w=label.shape
#             pred = pred.cpu().numpy()#转换成数组，方便循环
#             pred=np.asarray(np.argmax(pred, axis=1), dtype=np.uint8)#变成了三维
#             pred = np.reshape(pred, (b,1,h,w))
#             label=label.cpu().numpy()#仍然是思维
#             assert pred.shape==label.shape
#             for i in range(b):
#                 label_i = label[i, 0, :, :]
#                 if(np.all(label_i==ignore_label)):
#                     continue
#                 pred_i = pred[i, 0, :, :]
#                 filter = label_i != ignore_label
#                 label_i = label_i[filter]
#                 pred_i = pred_i[filter]
#                 #计算洪水这个类别的iou
#                 tpi = np.sum((label_i == 1) & (pred_i == 1))
#                 fni = np.sum((label_i == 0) & (pred_i == 1))
#                 fpi = np.sum((label_i == 1) & (pred_i == 0))
#                 tni = np.sum((label_i == 0) & (pred_i == 0))
#                 tni_sum.append(tni)
#                 fpi_sum.append(fpi)
#                 fni_sum.append(fni)
#                 tpi_sum.append(tpi)
#                 if tpi==0:
#                     Ioui=0
#                 else:
#                     Ioui=tpi/(fni+fpi+tpi)
#                 mIou_.append(Ioui)
#                 accuracyi = (tpi+tni)/(tpi+fpi+tni+fni)
#                 accuracy.append(accuracyi)
#             if sv_pred:
#                 sv_path = os.path.join(sv_dir, 'test_results')
#                 if not os.path.exists(sv_path):
#                     os.mkdir(sv_path)
#                 test_dataset.save_pred(pred, sv_path,name)
#             if index % 5 == 0:# 这一段就是方便阅读，没什么实际意义，可以不用看
#                 logging.info('processing: %d images' % index)
#     IoU=np.mean(mIou_)
#     mIoU=np.sum(tpi_sum)/(np.sum(fni_sum)+np.sum(fpi_sum)+np.sum(tpi_sum))
#     omission=np.sum(fni_sum)/(np.sum(fni_sum)+np.sum(tpi_sum))#漏保
#     commission=np.sum(fpi_sum)/(np.sum(fpi_sum)+np.sum(tni_sum))#误报
#     OA=np.mean(accuracy)
#     return mIoU,IoU,omission,commission,OA
# def validate(config, testloader, model, writer_dict,num_class,ignore_label,train_dataset):
#     model.eval()
#     mIou_=[]
#     tpi_sum=[]
#     tni_sum=[]
#     fpi_sum=[]
#     fni_sum=[]
#     accuracy=[]
#     with torch.no_grad():
#         for idx, batch in enumerate(testloader):
#             image, size,labels,name = batch
#             size = labels.size()
#             # print(f"size:{size}")#size:torch.Size([8, 1, 512, 512])
#             image = image.float().cuda()
#             labels = labels.long().cuda()
#             losses,loss0, preds = model(image, labels)
#             b,c,h,w=preds.shape
#             preds = preds.cpu().numpy()#转换成数组，方便循环
#             preds=np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)#变成了三维
#             preds = np.reshape(preds, (b,1,h,w))
#             label=labels.cpu().numpy()#仍然是思维
#             assert preds.shape==label.shape
#             for i in range(b):
#                 label_i = label[i, 0, :, :]
#                 if(np.all(label_i==ignore_label)):
#                     continue
#                 pred_i = preds[i, 0, :, :]
#                 filter = label_i != ignore_label
#                 label_i = label_i[filter]
#                 pred_i = pred_i[filter]
#                 #计算洪水这个类别的iou
#                 tpi = np.sum((label_i == 1) & (pred_i == 1))
#                 # print(tpi)
#                 tpi_sum.append(tpi)
#                 fni = np.sum((label_i == 0) & (pred_i == 1))
#                 # print(fni)
#                 fni_sum.append(fni)
#                 fpi = np.sum((label_i == 1) & (pred_i == 0))
#                 # print(fpi)
#                 fpi_sum.append(fpi)
#                 tni = np.sum((label_i == 0) & (pred_i == 0))
#                 # print(tni)
#                 tni_sum.append(tni)
#                 accuracyi = (tpi+tni)/(tpi+fpi+tni+fni)
#                 accuracy.append(accuracyi)
#                 if tpi==0:
#                     Ioui=0
#                 else:
#                     Ioui=tpi/(fni+fpi+tpi)
#                 mIou_.append(Ioui)
#             if idx % 5 == 0:
#                 print(idx)
#     IoU=np.mean(mIou_)
#     mIoU=np.sum(tpi_sum)/((np.sum(fni_sum)+np.sum(fpi_sum)+np.sum(tpi_sum)))
#     omission=np.sum(fni_sum)/(np.sum(fni_sum)+np.sum(tpi_sum))#漏保
#     commission=np.sum(fpi_sum)/(np.sum(fpi_sum)+np.sum(tni_sum))#误报
#     OA=np.mean(accuracy)
#     return mIoU,IoU,omission,commission,OA
# def test(config, test_dataset, testloader, model,
#          sv_dir='./', sv_pred=True):
#     model.eval()
#     with torch.no_grad():
#         for idx, batch in enumerate(testloader):
#             image, size,label = batch
#             pred = test_dataset.single_scale_inference(
#                 config,
#                 model,
#                 image.cuda())
#
#             if sv_pred:
#                 sv_path = os.path.join(sv_dir,'test_results')
#                 if not os.path.exists(sv_path):
#                     os.mkdir(sv_path)
#                 test_dataset.save_pred(pred, sv_path)
