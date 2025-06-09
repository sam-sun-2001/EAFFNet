# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

class BasicBlock(nn.Module):
    """
    通过两个卷积层来实现特征提取和降维度，默认inplanes=planes*expansion
    """
    expansion = 1 #通道扩展的倍数
    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        """
        残差块
        :param inplanes: 输入图的通道数
        :param planes: 经过残差块之后，输出图的通道数
        :param stride: int，为1.
        :param downsample: None
        :param no_relu: 连接之后，是否经过relu函数
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)#大小不变
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)#不变
        self.relu = nn.ReLU(inplace=True)#不变
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)#不变
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)#不变
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)#size~
        out = self.bn1(out)#size~
        out = self.relu(out)#size~
        out = self.conv2(out)#size~
        out = self.bn2(out)#size~ 通道数从inplanes变成了planes
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual#相加，需要通道数一样才可以相加
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Module):
    """
    通过三个卷积层来实现特征提取和降维,默认不改变特征图的高度和宽度
    """
    expansion = 2 #通道扩展的倍数，用于增加输出通道
    # final real output planes should be double planes
    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)



class segmenthead(nn.Module):
    """
    接受输入特征图，输出语义分割的最终预测结果
    segmenthead 类通常用于语义分割网络的最后阶段，生成每个像素的分类结果。
    """

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        """
        接受输入特征图，输出语义分割的最终预测结果
        :param inplanes:输入通道数
        :param interplanes:中间层的通道数
        :param outplanes:输出通道谁
        :param scale_factor:用于上采样的比例因子（optional）
        """

        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        # out: x->bn1->relu->conv1->bn2->relu->conv2
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        if self.scale_factor is not None:#双线性插值上采样到指定大小
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)

        return out

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(DAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out 
"""
PAPPM有什么作用？为什么很多研究论文都用了它？
"""
class PAPPM(nn.Module):
    """
    输入：N，c，h,w
    """
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        '''
        inplane: 输入通道数  64
        branch_planes：分支通道数  100
        outplanes：输出通道数

        池化
        '''
        super(PAPPM, self).__init__()
        bn_mom = 0.1


        #输入：6,64,1024,2048
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
        #N,branch_planes,513,1024                            #N,inplanes,513, 1024
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    #N,inplanes,513, 1024
                                    nn.ReLU(inplace=True),
                                    #N,inplanes,513, 1024
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    #N,branch_planes,513,1024
                                    )

        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
        #N,branch_planes,256,511                            #N,inplanes,256,511
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    #N,inplanes,256,511
                                    nn.ReLU(inplace=True),
                                    #N,inplanes,256,511
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    #N,branch_planes,256,511
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
        #N,branch_planes,129,257                            #N,inplanes,129,257
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    #N,inplanes,129,257
                                    nn.ReLU(inplace=True),
                                    #N,inplanes,129,257
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    #N,branch_planes,129,257
                                    )



        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), #自适应平均池化操作后
        #N,brach_planes,1,1                            #N,inplanes,1,1
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    #N,inplanes,1,1
                                    nn.ReLU(inplace=True),
                                    #N,inplanes,1,1
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    #N,brach_planes,1,1
                                    )

        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    #N,inplanes,1024,2048
                                    nn.ReLU(inplace=True),
                                    #N,inplanes,1024,2048
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    ##N,branch_planes,1024,2048
                                    )


        #共有5个尺度scale0，1，2，3，4
        """
        0:N,branch_planes,1024,2048  原尺寸         ，扩大通道数
        1:N,branch_planes,513,1024   减小尺寸，      扩大通道数
        2:N,inplanes,256,511         减小尺寸，       原通道数
        3:N,inplanes,129,257         减小尺寸，       原通道数
        4:N,brach_planes,1,1         减小尺寸        扩大通道数
        5:scale_process              尺寸不变        通道数不变
        6：compression                尺寸不变       通道数变
        7：shortcut                  尺寸不变         通道数变
        """
        #分组卷积
        self.scale_process = nn.Sequential(
                                    BatchNorm(branch_planes*4, momentum=bn_mom),#N，branch_planes*4，1024,2048
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes*4, branch_planes*4, kernel_size=3, padding=1, groups=4, bias=False),
            #分组卷积，指的是 输入一个通道为N的输入，将其通道分成G组，每一组的通道数为N/G,每一组进行普通卷积
                                    )

      
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):
        #获取图片的长和宽
        # x->scale0
        width = x.shape[-1]
        height = x.shape[-2]        
        scale_list = []
        x_ = self.scale0(x) #原尺寸扩大通道数
        scale_list.append(F.interpolate(self.scale1(x), size=[height, width],   #x->scale1->up->+x_=x1
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale2(x), size=[height, width],   #x->scale2->up->+x_
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale3(x), size=[height, width],   #x->scale3->up->+x_
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width],   #x->scale4->up->+x_
                        mode='bilinear', align_corners=algc)+x_)

        scale_out = self.scale_process(torch.cat(scale_list, 1)) #沿着通道维度进行拼接，拼接结束的形状是：N,branch_planes*4,1024,2048
        out = self.compression(torch.cat([x_,scale_out], 1)) + self.shortcut(x)
        return out
    

class PagFM(nn.Module):
    """
    来选择性地融合来自两个不同分支的特征
    """
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        """

        @param in_channels:输入特征图的通道数。
        @param mid_channels:中间层特征图的通道数。
        @param after_relu: 一个布尔值，表示是否在处理特征图之前应用ReLU激活函数。
        @param with_channel:一个布尔值，表示是否在计算注意力图时使用通道信息
        @param BatchNorm:批归一化层的类型，默认是 nn.BatchNorm2d
        """
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        #f_x 和 f_y：并行的卷积和批归一化序列，用于处理来自两个不同分支的特征图。
        self.f_x = nn.Sequential(#卷积+批归一化： in_channel->mid_channel
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        self.f_y = nn.Sequential(#卷积+批归一化: in_channel->mid_channel
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        if with_channel:# 如果在计算注意力图时，使用通道信息
            self.up = nn.Sequential(#一个卷积和批归一化序列，用于调整通道维度。 mid—>in_channels
                                    nn.Conv2d(mid_channels, in_channels, 
                                              kernel_size=1, bias=False),
                                    BatchNorm(in_channels)
                                   )
        if after_relu:#激活 relu
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        """
        注意尺寸的
        @param x: 来自P分支的特征退
        @param y: 来自D分支的特征图
        @return:
        """
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        
        y_q = self.f_y(y)#卷积+批归一化: y_q的大小跟y的大小一样
        #让y_q的形状跟x一样
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        ##y_q的形状跟x一样
        x_k = self.f_x(x)
        #x_k的形状跟x的形状一样
        #y_k的形状跟x_k的形状一样
        #例如： 两个都是【3，20，30，40】
        
        if self.with_channel:
            #计算相似度
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        #让y跟x一样
        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x = (1-sim_map)*x + sim_map*y
        return x
    
class Light_Bag(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Light_Bag, self).__init__()
        self.conv_p = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )
        self.conv_i = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )
        
    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        
        p_add = self.conv_p((1-edge_att)*i + p)
        i_add = self.conv_i(i + edge_att*p)
        
        return p_add + i_add
    

class DDFMv2(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(DDFMv2, self).__init__()
        self.conv_p = nn.Sequential(
                                BatchNorm(in_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )
        self.conv_i = nn.Sequential(
                                BatchNorm(in_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )
        
    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        
        p_add = self.conv_p((1-edge_att)*i + p)
        i_add = self.conv_i(i + edge_att*p)
        
        return p_add + i_add

class Bag(nn.Module):
    """
    它的主要功能是利用边缘注意力机制在两个输入特征图之间进行加权融合，并通过一个卷积层进行处理
    """
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Bag, self).__init__()

        self.conv = nn.Sequential(
                                BatchNorm(in_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=3, padding=1, bias=False)                  
                                )

        
    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att*p + (1-edge_att)*i)


if __name__ == '__main__':

    
    x = torch.rand(4, 64, 32, 64).cuda()
    y = torch.rand(4, 64, 32, 64).cuda()
    z = torch.rand(4, 64, 32, 64).cuda()
    net = PagFM(64, 16, with_channel=True).cuda()
    
    out = net(x,y)