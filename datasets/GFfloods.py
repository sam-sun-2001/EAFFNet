# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import random
import os
import rasterio
import cv2
import numpy as np
from PIL import Image

import torch
from .base_dataset import BaseDataset

# 训练数据：train_split
# 验证数据： flood_valid_data.lst

#以训练数据为例，写下面的代码逻辑
class GFfloods(BaseDataset):
    def __init__(self,
                 root,
                 type,#train
                 list_path,
                 flip,
                 crop_size,
                 num_classes=2,
                 ignore_label=-1,
                 base_size=512,
                 mean=[0.6851, 0.5235],
                 std=[0.0820, 0.1102]
                 ):
        super(GFfloods, self).__init__(ignore_label, base_size,
                                       mean, std,)

        self.root = root
        self.type=type
        self.list_path = list_path
        self.num_classes = num_classes
        self.flip=flip
        self.crop_size=crop_size
        self.img_list = [(line.strip().split('.tif ')[0] + '.tif', line.strip().split('.tif ')[1]) for line in open(list_path)]


        self.dataItemLis = self.read_files()

        #原始数据的标签映射
        self.label_mapping = {
            0: 0,  # NoData
            255: 0,              # Not Water
            1: 1               # Water
        }
        self.class_weights = torch.FloatTensor([0.8373, 1.1529]).cuda()



    def read_files(self):
        """把train。list，以及test。list两个文本文件给读到字典里面去
        """
        dataItemLis = []
        if self.type=="test": #验证数据集
            for item in self.img_list:
                image_path,label_path = item
                dataItemLis.append({
                    "img": image_path,
                    "label": label_path
                })
        else: #train
            for item in self.img_list:
                image_path, label_path = item
                dataItemLis.append({
                    "img": image_path,
                    "label": label_path
                })
        return dataItemLis

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label.astype(np.int64)

    def input_transform(self, image):
        """
        进行归一化
        :param image:1,256,256
        :return:
        """
        # 将图像转换为浮点数格式
        image = image.astype(np.float32)
        # 使用指定的均值和标准差进行归一化
        mean = np.array([410,299,232,286,338])[:, None, None]
        std = np.array([37,38,45,49,133])[:, None, None]
        image = (image - mean) / std
        return image.astype(np.float32)


    def rand_crop(self, image, label, crop_size):
        if crop_size == 0:
            return image, label

        # print(f"image.shape: {image.shape}")
        _, h, w = image.shape  # 获取高度和宽度

        # 检查图像是否足够大以容纳裁剪区域
        if h < crop_size[0] or w < crop_size[1]:
            raise ValueError("图像尺寸小于裁剪尺寸，请确保图像至少与裁剪尺寸相等。")

        # 生成随机裁剪的起点，以免超出边界
        x = random.randint(0, w - crop_size[1])
        y = random.randint(0, h - crop_size[0])

        # 裁剪图像和标签
        image = image[:, y:y + crop_size[0], x:x + crop_size[1]]
        label = label[:, y:y + crop_size[0], x:x + crop_size[1]]

        return image, label



    def flip2(self, im, label):
        """
        随机对输入图像和标签进行水平或垂直翻转。

        参数:
        - im: 三维图像张量，形状为 (channels, height, width)
        - label: 三维标签张量，形状为 (height, width)

        返回:
        - 翻转后的图像和标签，形状不变
        """
        # 随机选择翻转类型
        flip_type = np.random.choice(["horizontal", "vertical"])

        if flip_type == "horizontal":
            # 水平翻转 (沿宽度方向翻转)
            return im[:, :, ::-1], label[:, :, ::-1]
        elif flip_type == "vertical":
            # 垂直翻转 (沿高度方向翻转)
            return im[:, ::-1, :], label[:, ::-1, :]

    def data_agument(self,image,label,crop_size,flip):
        """
        定义数据增强函数，用到了flip和crop函数
        :return: 返回经过crop和flip之后的照片和标签
        """
        # image,label=self.rand_crop(image,label,crop_size)
        if flip==True:
            image,label=self.flip2(image,label)
            return image,label
        else:
            return image,label



    def __getitem__(self, index):
        """拿数据"""
        item = self.dataItemLis[index]
        img=self.root+item["img"]
        name=item["img"]
        with rasterio.open(img) as src:
            image = src.read()# 读取图片：返回的是numpy类型的数组
            # print(image.shape)5,256,256
        size = image.shape#
        # print(size)
        if self.type=="test":#测试的数据集
            image = self.input_transform(image)
            if np.isnan(image).any():
                print("有nan")
                image = np.nan_to_num(image, nan=0.0)  # 将 NaN 替换为 0
            label_path="/root/autodl-tmp/pycharm_project_666/data/GFfloodsnet/all_train_label/"+item["label"]
            with rasterio.open(label_path) as src:# 读取标签
                label = src.read()
                label = self.convert_label(label)
            return image.copy(), np.array(size), label,name
        #训练数据集
        if np.isnan(image).any():
            print("有nan")
            image = np.nan_to_num(image, nan=0.0)  # 将 NaN 替换为 0
        label_path="/root/autodl-tmp/pycharm_project_666/data/GFfloodsnet/all_train_label/"+item["label"]
        with rasterio.open(label_path) as src:# 读取标签
            label = src.read()#int,16
        label = self.convert_label(label)
        # print(f"image.shape:{image.shape}")
        image,label=self.data_agument(image,label,crop_size=self.crop_size,flip=self.flip)
        # print(f"hih:{image.shape}")
        image=self.input_transform(image)
        # print(f"image.copy:{image.copy().shape}")
        # print(label.copy().shape)
        return image.copy(), label.copy() #edge.copy(), np.array(size)


    def single_scale_inference(self, config, model, image):
        """
        进行推理
        @return: 得分
        """
        model=model.getModel()
        pred = model(image)#作者自己写的父类方法
        return pred


    def save_pred(self, preds, sv_path, name):
        """
        将预测的结果保存成图像
        preds： 得分！！ 如，6,2,512,512
        sv_path：测试照片可视化的目录
        name:保存照片的名称？
        """

        if not os.path.exists(sv_path):
            os.makedirs(sv_path)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)

        for i in range(preds.shape[0]):
            pred=preds[i]
            #创建一个RGB图像数组
            rgb_image=np.zeros((pred.shape[0],pred.shape[1],3),dtype=np.uint8)
            #设置预测对应的颜色
            rgb_image[pred==1]=[255,255,255]
            rgb_image[pred==0]=[0,0,0]
            save_img = Image.fromarray(rgb_image)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))



