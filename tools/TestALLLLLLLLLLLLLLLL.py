import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools.dataset import myDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def chart(mask, pred, nodata=-1):
    """
    mask: 真实的标签, np.ndarray, 形状为[height, width], 其中-1表示忽略的像素
    pred: 预测的标签, np.ndarray, 形状为[height, width], 其中0和1分别表示两个不同的类别
    """
    # 忽略标签值为-1的像素
    index = mask != nodata
    # 计算交集和并集
    tp = np.logical_and(mask*index, pred*index).sum()
    # 计算假阳性（False Positive）：预测为1，但真实标签为0
    fp = np.logical_and(pred == 1, mask == 0).sum()
    # 计算假阴性（False Negative）：预测为0，但真实标签为1
    fn = np.logical_and(pred == 0, mask == 1).sum()
    #tn
    tn = np.logical_and(pred == 0, mask == 0).sum()

    union = np.logical_or(mask*index, pred*index).sum()

    if union == 0:
        iou=0
    else:
        iou=(tp+0.000001) / (union+0.000001)
    if (tp+fp)==0:
        precision=0
    else:
        precision=(tp+0.000001)/(tp+fp+0.000001)
    if (tp+fn)==0:
        recall=0
    else:
        recall=(tp+0.000001)/(tp+fn+0.000001)
    if (precision+recall==0):
        f1=0
    else:
        f1=2*((precision*recall+0.000001)/(precision+recall++0.000001))
    if (tp+tn+fp+fn)==0:
        OA=0
    else:
        OA=(tp+tn+0.000001)/(tp+tn+fp+fn+0.000001)
    return OA,iou,precision,recall,f1


def cal_iou(pred, target, pro = 0.5):
    """
    进行了一个预测
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

    validOA,iou,precision,recall,f1 = chart(target, pred, nodata=255)

    return validOA,iou,precision,recall,f1


def valid_model(model,  valid_dataload):
    """
    验证精度
    """
    # validation, 这一部分用来查找最优的训练模型
    validIOU = 0.0
    validPrecision = 0.0
    validRecall = 0.0
    validF1 = 0.0
    validOA = 0.0
    model.eval()

    with torch.no_grad():
        step = 0
        for Img, Lbl in valid_dataload:
            step = step + 1

            Img = Img.to(device=device, dtype=torch.float)
            Lbl = Lbl.to(device=device, dtype=torch.long)
            outputs = model(Img)

            # 计算IOU精度
            oa,iou,precision,recall,f1 = cal_iou(outputs, Lbl)
            validIOU = validIOU + iou
            validPrecision=validPrecision+precision
            validRecall=validRecall+recall
            validF1=validF1+f1
            validOA=validOA+oa

    if step == 0:
        return validOA,validIOU,validPrecision,validRecall,validF1
    else:
        validOA /= step
        validIOU /= step
        validPrecision /= step
        validRecall /= step
        validF1 /= step

        return validOA,validIOU,validPrecision,validRecall,validF1



# import random
# print('Seeding with', 304)
# random.seed(304)
# torch.manual_seed(304)
from models.unet import Unet
model=Unet(n_channels=2,n_classes=2).to(device=device)
#加载权重
final_output_dir='/root/autodl-tmp/pycharm_project_666/_r1_checkpoints/weights_60_0.40.pth' #pth
if os.path.isfile(final_output_dir):
    checkpoint = torch.load(final_output_dir)
    dct = checkpoint['model']#获取模型的权重！！！！！！！！
    model.load_state_dict(dct)#使用load_state_dict加载模型权重
test_dataset = myDataset("/root/autodl-tmp/pycharm_project_666/data/Sen1Floods11/flood_test_data.csv",0, type="S1", batchsize=4)
test_dataloaders = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last = True)
validOA,validIOU,validPrecision,validRecall,validF1=valid_model(model,test_dataloaders)
print(f"validOA:{validOA},validIoU:{validIOU},validPrecision:{validPrecision},validRecall:{validRecall},validF1:{validF1}")