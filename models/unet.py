#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # class GFM(nn.Module):
# #     def __init__(self, inchannel, midchannel, outchannel):
# #         super(GFM, self).__init__()
# #         # 分配权重，将其定义为可训练参数
# #         self.weight = nn.Parameter(torch.tensor([0.65, 0.5]))  # 0.65:细节部分的权重
# #         # 定义网络层
# #         self.sig = nn.Sigmoid()
# #         self.sig1 = nn.Sigmoid()
# #         self.bn1=nn.BatchNorm2d(midchannel)
# #         self.bn2=nn.BatchNorm2d(outchannel)
# #         self.conv1 = nn.Conv2d(in_channels=inchannel, out_channels=midchannel, kernel_size=3, stride=1, padding=1)
# #         self.conv2 = nn.Conv2d(in_channels=midchannel * 2, out_channels=outchannel, kernel_size=3, stride=1, padding=1)
# #
# #     def forward(self, x1, x2):
# #         # x1: 细节信息，赋予更高的权重
# #         # x2: 语义信息，赋予较低的权重
# #         # print(f"x1:{x1.shape}")[8, 64, 256, 256]
# #         # print(f"x2:{x2.shape}")[8, 64, 256, 256]
# #         # 计算加权组合
# #         x_ = x1 * self.weight[0] + x2 * self.weight[1]
# #         x_ = self.conv1(x_)
# #         x_=self.bn1(x_)
# #         x_ = self.sig(x_)
# #         # 计算逐元素乘积
# #         x1_ = x_ * x1
# #         x2_ = (1 - x_) * x2
# #         # 拼接张量 (需要在 `dim=1` 维度拼接，因为这是通道维度)
# #         y = torch.cat((x1_, x2_), dim=1)
# #         # 通过第二个卷积层和激活函数
# #         y = self.sig1(self.bn2(self.conv2(y)))
# #         return y
from torch import nn


class GFM(nn.Module):
    """
    这个是原论文的GFM模块
    """
    def __init__(self,inchannel,outchannel):
        super(GFM, self).__init__()
        self.conv1X1=nn.Conv2d(in_channels=2*inchannel,out_channels=inchannel,kernel_size=1)
        self.bn1=nn.BatchNorm2d(inchannel)
        self.sig=nn.Sigmoid()
        self.conv1X1_=nn.Conv2d(in_channels=2*inchannel,out_channels=outchannel,kernel_size=1)
        self.bn2=nn.BatchNorm2d(outchannel)
        self.sig2=nn.Sigmoid()
    def forward(self,x1,x2):
        """
        :param x1: optical
        :param x2: sar
        :return:
        """
        G=torch.concat((x1,x2),dim=1)
        G=self.conv1X1(G)
        G=self.bn1(G)
        G=self.sig(G)
        x1_=x1 * G
        x2_=x2 * (1-G)
        output=torch.concat((x1_,x2_),dim=1)
        output=self.conv1X1_(output)
        output=self.bn2(output)
        output=self.sig2(output)
        return output
# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.double_conv(x)
#
# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )
#
#     def forward(self, x):
#         return self.maxpool_conv(x)
#
# class Up(nn.Module):#self.up4 = Up(128, 64, bilinear)
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super(Up, self).__init__()
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)
#
#     def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#         x1 = self.up(x1)
#         # Calculate padding to align x1 and x2
#         diff_y = x2.size()[2] - x1.size()[2]
#         diff_x = x2.size()[3] - x1.size()[3]
#         x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
#                         diff_y // 2, diff_y - diff_y // 2])
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)
#
# class Up2(nn.Module):# in_channels=128,out_channels=64
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super(Up2, self).__init__()
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv1=nn.Conv2d(kernel_size=1,in_channels=128,out_channels=64)#调整通道数目
#             self.conv = DoubleConv(in_channels=64, out_channels=32, mid_channels=32)
#             self.gfm=GFM(inchannel=in_channels // 2,outchannel=out_channels)
#
#             self.bn1=nn.BatchNorm2d(64)
#             self.sig=nn.Sigmoid()
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv1=nn.Conv2d(kernel_size=1,in_channels=128,out_channels=64)#调整通道数目
#             self.conv = DoubleConv(in_channels=64, out_channels=32, mid_channels=32)
#             self.gfm=GFM(inchannel=in_channels // 2,outchannel=out_channels)
#             self.bn1=nn.BatchNorm2d(64)
#             self.sig=nn.Sigmoid()
#
#     def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#         # print(f"x1:{x1.shape}")x1:torch.Size([8, 128, 128, 128])
#         # print(f"x2:{x2.shape}")x2:torch.Size([8, 64, 256, 256])
#         x1 = self.up(x1)
#         # print(f"x1:{x1.shape}")1:torch.Size([8, 128, 256, 256])
#         diff_y = x2.size()[2] - x1.size()[2]
#         # print(diff_y)  0
#         diff_x = x2.size()[3] - x1.size()[3]
#         # print(diff_x)  0
#         x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
#                         diff_y // 2, diff_y - diff_y // 2])
#         #这俩通道维度不一样
#         x1=self.sig(self.bn1(self.conv1(x1)))
#         # print(f"x1!!!!!!!!!!:{x1.shape}")
#         x=self.gfm(x1,x2)  #x1:torch.Size([8, 64, 256, 256]),x2也是，一样
#         # print(f"x:{x.shape}")#x:torch.Size([8, 64, 256, 256])
#         x=self.conv(x)
#         return x
# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         return self.conv(x)
#
# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 1024)
#         factor = 2 if bilinear else 1
#         self.double_conv = DoubleConv(1024, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         # self.up4 = Up(128, 64, bilinear)
#         self.up4=Up2(in_channels=128,out_channels=64)
#         self.outc = OutConv(32, n_classes)
#         self.initialize_weights()
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x5 = self.double_conv(x5)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits
#     def initialize_weights(self):
#         # 遍历模型中的每一层，初始化权重
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
# if __name__ == '__main__':
#     net = UNet(n_channels=2, n_classes=2).cuda()
#     x_input = torch.rand(2, 2, 512, 512).cuda()  # 如果你需要一个形状为 (1, 1, 512, 512) 的随机张量
#     y_output=net(x_input)
#     print(y_output.shape)
#
#
#

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):#self.up4 = Up(128, 64, bilinear)
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Calculate padding to align x1 and x2
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# class Up2(nn.Module):# in_channels=128,out_channels=64
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super(Up2, self).__init__()
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv1=nn.Conv2d(kernel_size=1,in_channels=128,out_channels=64)#调整通道数目
#             self.conv = DoubleConv(in_channels=64, out_channels=32, mid_channels=32)
#             self.gfm=GFM(inchannel=in_channels // 2,outchannel=out_channels)
#
#             self.bn1=nn.BatchNorm2d(64)
#             self.sig=nn.Sigmoid()
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv1=nn.Conv2d(kernel_size=1,in_channels=128,out_channels=64)#调整通道数目
#             self.conv = DoubleConv(in_channels=64, out_channels=32, mid_channels=32)
#             self.gfm=GFM(inchannel=in_channels // 2,midchannel=in_channels // 2,outchannel=out_channels)
#             self.bn1=nn.BatchNorm2d(64)
#             self.sig=nn.Sigmoid()
#
#     def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#         # print(f"x1:{x1.shape}")x1:torch.Size([8, 128, 128, 128])
#         # print(f"x2:{x2.shape}")x2:torch.Size([8, 64, 256, 256])
#         x1 = self.up(x1)
#         # print(f"x1:{x1.shape}")1:torch.Size([8, 128, 256, 256])
#         diff_y = x2.size()[2] - x1.size()[2]
#         # print(diff_y)  0
#         diff_x = x2.size()[3] - x1.size()[3]
#         # print(diff_x)  0
#         x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
#                         diff_y // 2, diff_y - diff_y // 2])
#         #这俩通道维度不一样
#         x1=self.sig(self.bn1(self.conv1(x1)))
#         # print(f"x1!!!!!!!!!!:{x1.shape}")
#         x=self.gfm(x1,x2)  #x1:torch.Size([8, 64, 256, 256]),x2也是，一样
#         # print(f"x:{x.shape}")#x:torch.Size([8, 64, 256, 256])
#         x=self.conv(x)
#         return x
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        factor = 2 if bilinear else 1
        self.double_conv = DoubleConv(1024, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64// factor, bilinear)
        # self.up4=Up(in_channels=128,out_channels=64)
        self.outc = OutConv(64, n_classes)
        self.initialize_weights()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.double_conv(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    def initialize_weights(self):
        # 遍历模型中的每一层，初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)




