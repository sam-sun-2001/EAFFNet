from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

#Nested Unet

class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, in_ch=5, out_ch=2):
        super(NestedUNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]#64,128,256,512,1024

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])#2->64
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])#64->128
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])#512->1024

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])#192->64
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])#128+256->128
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])#256+512->256
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])#512+1024->512

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])#
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.initialize_weights()

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


    def forward(self, x):

        x0_0 = self.conv0_0(x) #x1
        x1_0 = self.conv1_0(self.pool(x0_0))#x2
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))#yansuo:64

        x2_0 = self.conv2_0(self.pool(x1_0))#x3
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))#yansuo:128
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))#x1,yansuo64,yansuo128->64

        x3_0 = self.conv3_0(self.pool(x2_0))#x4
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))#x3,x4->256
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))#x5
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))
        output = self.final(x0_4)
        return output

if __name__=="__main__":
    net=NestedUNet(in_ch=5,out_ch=2)
    inp=torch.rand(2,5,256,256)
    out=net(inp)
    print(out.shape)







