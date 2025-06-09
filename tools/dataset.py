# coding:utf-8
import models.ImageRead as ImageRead
from torch.utils.data import Dataset
import os

import torch
import numpy as np
import csv
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hcf(numSample, batchsize):
    isBreak = 1
    while (isBreak):
        if numSample % batchsize == 0:
            isBreak = 0
        else:
            numSample = numSample - 1

    return numSample



class Transform():
    """
    数据增强功能
    """
    def __init__(self, img_data, label_data):
        self.img_data = img_data
        self.label_data = label_data

    def RandomHorizentalFlip(self, prob):  # 水平翻转
        if random.random() < prob:
            self.img_data = self.img_data[:, ::-1]
            self.label_data = self.label_data[:, ::-1]
        return self.img_data, self.label_data

    def RandomVertialFlip(self, prob):
        if random.random() < prob:
            self.img_data = self.img_data[::-1]
            self.label_data = self.label_data[::-1]
        return self.img_data, self.label_data

    def RandomRotate90(self, prob):        # 按对角线翻转
        if random.random() < prob:
            self.img_data = self.img_data.swapaxes(1, 0)
            self.img_data = self.img_data[:, ::-1]
            self.label_data = self.label_data.swapaxes(1, 0)
            self.label_data = self.label_data[:, ::-1]
        return self.img_data, self.label_data

    def RandomRotate(self, prob):          # 按对角线翻转
        if random.random() < prob:
            self.img_data = self.img_data.swapaxes(1, 0)
            self.label_data = self.label_data.swapaxes(1, 0)

        return self.img_data, self.label_data


def make_dataset(CSVPath, type="S1"):
    """
    获取图像和标签
    """
    labeFileList = []

    if type == "S1":
        FilePath = os.path.join(os.path.dirname(CSVPath), 'all_train_data')
        labelPath = os.path.join(os.path.dirname(CSVPath), 'all_train_label')
    elif type == "S2":
        FilePath = os.path.join(os.path.dirname(CSVPath), 'all_train_data')
        labelPath = os.path.join(os.path.dirname(CSVPath), 'all_train_label')

    with open(CSVPath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) > 1:
                labeFileList.append(row[1])

    labelImgPathList = []
    ImgPathList = []

    for labelFileName in labeFileList:
        if type == "S1":
            imgFileName = labelFileName.replace("Label", "S1")

            if os.path.exists(os.path.join(FilePath, imgFileName)) and os.path.exists(os.path.join(labelPath, labelFileName)) and ('tif' in imgFileName):
                ImgPathList.append(os.path.join(FilePath, imgFileName))
                labelImgPathList.append(os.path.join(labelPath, labelFileName))

        elif type == "S2":
            imgFileName = labelFileName.replace("Label", "S2")

            if os.path.exists(os.path.join(FilePath, imgFileName)) and os.path.exists(os.path.join(labelPath, labelFileName)) and ('tif' in imgFileName):
                ImgPathList.append(os.path.join(FilePath, imgFileName))
                labelImgPathList.append(os.path.join(labelPath, labelFileName))

    return labelImgPathList, ImgPathList


class myDataset(Dataset):
    def __init__(self, CSVPath,dataAug, type="S1",batchsize=2):

        labelImgPathList, ImgPathList = make_dataset(CSVPath, type)

        numImg = len(labelImgPathList)
        numTrain = hcf(numImg, batchsize)

        self.Imgs_Path = ImgPathList[:numTrain]

        self.label_Path = labelImgPathList[:numTrain]

        self.type = type

        self.dataAug = dataAug


    def __getitem__(self, index):
        t1_path = self.Imgs_Path[index]
        # print(t1_path)

        t1_label_path = self.label_Path[index]

        img_t1 = ImageRead.readimage(t1_path)

        img_t1 = np.transpose(img_t1, [1, 2, 0]).astype(np.float32)   # h, w, c

        img_t1Label = ImageRead.readimage(t1_label_path)

        img_t1Label[img_t1Label == -1] = 255

        img_t1Label = np.array(img_t1Label).astype(np.uint8)

        if self.type == "S1":
            img_t1 = (img_t1 + 30) / 30.0

        elif self.type == "S2":
            img_t1 = img_t1 / 10000.0


        if self.dataAug == 1:
            trans = Transform(img_t1, img_t1Label)
            methods = [trans.RandomHorizentalFlip(0.5), trans.RandomVertialFlip(0.5),
                       trans.RandomRotate90(0.5), trans.RandomRotate(0.5)]

            if random.random() < 0.5:                                 # 以50%的概率进行样本增强
                prob = random.randint(0, 3)
                img_t1, img_t1Label = methods[prob]


        img_t1 = np.transpose(img_t1, [2, 0, 1])
        img_t1 = np.ascontiguousarray(img_t1)
        img_t1 = np.nan_to_num(img_t1, nan=0)
        img_t1 = torch.from_numpy(img_t1).float()

        img_t1Label = np.ascontiguousarray(img_t1Label)
        img_t1Label = np.nan_to_num(img_t1Label, nan=0)
        img_t1Label = torch.from_numpy(img_t1Label).long()

        return img_t1, img_t1Label


    def __len__(self):
        return len(self.Imgs_Path)
