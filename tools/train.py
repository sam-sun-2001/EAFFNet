# -*- coding: utf-8 -*-
# train deep learning model
import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim

from models.focal import FocalLoss, DiceLoss
from tools.dataset import myDataset
from torch.nn import functional as F
import os
# from visdom import Visdom
import time
import numpy as np
import os.path
# from models.focal import FocalLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iou(mask, pred, nodata=-1):
    """
    mask: 真实的标签, np.ndarray, 形状为[height, width], 其中-1表示忽略的像素
    pred: 预测的标签, np.ndarray, 形状为[height, width], 其中0和1分别表示两个不同的类别
    """
    # 忽略标签值为-1的像素
    index = mask != nodata
    # 计算交集和并集
    intersection = np.logical_and(mask*index, pred*index).sum()
    union = np.logical_or(mask*index, pred*index).sum()
    # 计算IoU
    if union == 0:
        return 0
    else:
        return (intersection+0.000001) / (union+0.000001)


def cal_iou(pred, target, pro = 0.5):
    """
    计算MIOU
    pred, target: 预测和真实标签的NumPy数组, 形状为(h, w)
    num_classes: 类别数量, 默认为2
    """
    if torch.is_tensor(pred):
        pred = F.softmax(pred, dim=1)

        pred = pred.squeeze().cpu().numpy()
        pred = pred[1, :, :]

    if torch.is_tensor(target):
        target = target.squeeze().cpu().numpy()

    pred = pred > pro
    pred = np.array(pred).astype(np.uint8)

    IOU = iou(target, pred, nodata=255)

    return IOU


def valid_model(model, criterion1,criterion2, valid_dataload):
    """
    验证精度
    """
    # validation, 这一部分用来查找最优的训练模型
    validation_loss = 0.0
    validIOU = 0.0
    model.eval()

    with torch.no_grad():
        step = 0
        for Img, Lbl in valid_dataload:
            step = step + 1

            Img = Img.to(device=device, dtype=torch.float)
            Lbl = Lbl.to(device=device, dtype=torch.long)
            outputs = model(Img)
            # print(f"Validation logits range: min={outputs.min().item()}, max={outputs.max().item()}")
            loss1 = criterion1(outputs,Lbl)
            loss2 = criterion2(outputs,Lbl)
            loss=loss1+loss2
            validation_loss += loss.item()

            # 计算IOU精度
            IOU = cal_iou(outputs, Lbl)
            validIOU = validIOU + IOU

    if step == 0:
        return validation_loss, validIOU
    else:
        validation_loss /= step
        validIOU /= step

        return validation_loss, validIOU


def train_model(model, criterion1, criterion2,optimizer, train_dataloaders, valid_dataloaders, test_dataloaders,resume,final_output_dir, num_epochs=20):
    """
    train model
    """
    # viz = Visdom()  #
    # viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))
    last_epoch = 0
    global state, loss
    if resume==True:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')#存放checkpoint的路径
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            last_epoch = checkpoint['epoch']#获取epoch记录到哪一个epoch了
            dct = checkpoint['model']#获取模型的权重！！！！！！！！
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(dct)#使用load_state_dict加载模型权重
    for epoch in range(last_epoch+1,num_epochs):
        model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloaders.dataset)
        epoch_loss = 0
        step = 0
        for img_t1, img_t1Label in train_dataloaders:
            step += 1
            inputs_1 = img_t1.to(device=device, dtype=torch.float)
            # print(f"Validation logits range: min={inputs_1.min().item()}, max={inputs_1.max().item()}")
            img_t1Label = img_t1Label.to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            # forward
            img_t1_resutl = model(inputs_1.float())
            # print(f"Validation logits range: min={img_t1_resutl.min().item()}, max={img_t1_resutl.max().item()}")
            loss1 = criterion1(img_t1_resutl, torch.squeeze(img_t1Label).long())  # 解决1tensor错误问题
            loss2 = criterion2(img_t1_resutl, img_t1Label)  # 解决1tensor错误问题
            loss=loss1+(loss2)
            loss.backward()

            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()
            epoch_loss += loss.item()

            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloaders.batch_size + 1, loss.item()))

        print("epoch %d loss:%0.3f" % (epoch, epoch_loss / step))

        # validation
        validation_loss, validIOU = valid_model(model, criterion1,criterion2, valid_dataloaders)

        # test
        test_loss, testIOU = valid_model(model, criterion1, criterion2,test_dataloaders)

        writeData = "epoch " + str(epoch) +"  traini_loss:  "+str(epoch_loss / step)+ "  validIOU: " + str(validIOU) + " testIOU: " + str(testIOU)+ " validLoss: " + str(validation_loss) + " testLoss: " + str(test_loss)
        with open(saveIOUPath, 'a') as file_handle:  # 保存valid loss
            file_handle.write("{}\n".format(writeData))
        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch}
        torch.save(state, os.path.join(final_output_dir,'checkpoint.pth.tar'))
        if testIOU>0.40:
            state = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch}
            torch.save(state, os.path.join(final_output_dir,'weights_%d_0.40.pth' % epoch))
        elif testIOU>0.39:
            state = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch}
            torch.save(state, os.path.join(final_output_dir,'weights_%d_0.39.pth' % epoch))

        time.sleep(0.00001)

    torch.save(state['model'], os.path.join(final_output_dir,'last.pt'))
    return model


# 训练模型
def train(args, trainCSVPath, validCSVPath, testCSVPath, model, weight,resume,final_output_dir):

    batch_size = args.batch_size
    class_weight = torch.FloatTensor(weight).to(device=device, dtype=torch.float)
    criterion1 = nn.CrossEntropyLoss(weight=class_weight, ignore_index=255)
    # criterion1=FocalLoss(alpha=0.25, gamma=2.5, ignore_index=255, reduction='mean', weight=class_weight)
    criterion2=DiceLoss()

    learning_rate = args.init_learning_rate
    echo = args.echo

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # adam 优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.00005)  # adamW优化器

    train_dataset = myDataset(trainCSVPath,1, type="S1", batchsize=batch_size)            # 支持S1 和 S2
    train_dataloaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last = True)

    valid_dataset = myDataset(validCSVPath,0, type="S1", batchsize=batch_size)
    valid_dataloaders = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last = True)

    test_dataset = myDataset(testCSVPath,0, type="S1", batchsize=batch_size)
    test_dataloaders = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last = True)

    train_model(model, criterion1, criterion2,optimizer, train_dataloaders, valid_dataloaders, test_dataloaders,resume,final_output_dir, num_epochs=echo)


if __name__ == '__main__':

    # 参数解析
    import random
    print('Seeding with', 42)
    random.seed(42)
    torch.manual_seed(42)
    parse = argparse.ArgumentParser()

    parse.add_argument("--batch_size", type=int, help="batch size to train image ", default=4)
    parse.add_argument("--init_learning_rate", type=float, help="the init learning rate ", default=0.0003)
    parse.add_argument("--resume", type=bool, help="True if you want to resume", default=True)
    parse.add_argument("--echo", type=int, help="train echo ", default=120)
    parse.add_argument("--trainCSVPath", type=str, help="train image path ", default=
    "/root/autodl-tmp/pycharm_project_666/data/Sen1Floods11/flood_train_data.csv")
    parse.add_argument("--validCSVPath", type=str, help="train image path ", default=
    "/root/autodl-tmp/pycharm_project_666/data/Sen1Floods11/flood_valid_data.csv")
    parse.add_argument("--testCSVPath", type=str, help="train image path ", default=
    "/root/autodl-tmp/pycharm_project_666/data/Sen1Floods11/flood_test_data.csv")
    args = parse.parse_args()

    inputChannel = 2 # sentinel-2->13, sentinel-1->2
    numClass = 2

    final_output_dir='/root/autodl-tmp/pycharm_project_666/_rrrrrr3_checkpoints/'
    from models.unet import Unet

    model=Unet(n_channels=2,n_classes=2).to(device=device)

    weight = [0.05, 0.95]
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
    batch_size = args.batch_size
    resume=args.resume
    trainCSVPath = args.trainCSVPath
    validCSVPath = args.validCSVPath
    testCSVPath = args.testCSVPath
    saveIOUPath = os.path.join(final_output_dir,"iou.txt")


    train(args, trainCSVPath, validCSVPath, testCSVPath, model, weight,resume,final_output_dir)


