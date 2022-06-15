"""
使用了V5中的focus
使用了SPP结构
激活函数:
    3         -> v4   -> x
    LeakyReLU -> Mish -> SiLU(平滑的relu)
"""

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
from torch import nn

#---------------------------------------------------#
#   SiLU
#   nn.SiLU() or F.silu()
#---------------------------------------------------#
class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

#---------------------------------------------------#
#   获取激活函数
#---------------------------------------------------#
def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

#---------------------------------------------------#
#   conv + bn + 激活函数
#---------------------------------------------------#
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

#---------------------------------------------------#
#   深度可分离卷积
#---------------------------------------------------#
class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act,)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

#---------------------------------------------------#
#   Focus V5中使用了
#   一张图片每隔一个像素点拿一个值组成一张图片,最后可以让宽高变为原来一半,通道变为4倍  最后做一次卷积变换到合适维度
#   4x4x3 = 48 = 2x2x12=48
#---------------------------------------------------#
class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        #                  前边维度都要            2       3
        patch_top_left  = x[..., 0::2, 0::2]    # 0 2...  0 2...
        patch_bot_left  = x[..., 1::2, 0::2]    # 1 3...  0 2...
        patch_top_right = x[..., 0::2, 1::2]    # 0 2...  1 3...
        patch_bot_right = x[..., 1::2, 1::2]    # 1 3...  1 3...
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,)
        return self.conv(x)

#--------------------------------------------------#
#   残差结构的构建，小的残差结构
#   在这里通道和宽高都不变
#   两层卷积 1x1和3x3
#--------------------------------------------------#
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        super().__init__()
        hidden_channels = int(out_channels * expansion) # 中间通道减少一半
        Conv = DWConv if depthwise else BaseConv
        #--------------------------------------------------#
        #   利用1x1卷积进行通道数的缩减。缩减率一般是50%
        #--------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #--------------------------------------------------#
        #   利用3x3卷积进行通道数的拓张。并且完成特征提取
        #--------------------------------------------------#
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)

        # 只有使用残差且 in_channels == out_channels 时才使用残差模块
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

#--------------------------------------------------------------------#
#   CSPdarknet的结构块,每个stage的结构
#
#   CSPnet结构并不算复杂，就是将原来的残差块的堆叠进行了一个拆分，拆成左右两部分:
#   主干部分继续进行原来的残差块的堆叠；
#   另一部分则像一个残差边一样，经过少量处理直接连接到最后。
#   因此可以认为CSP中存在一个大的残差边。
#
#   V4中先进行了一次卷积让通道翻倍,宽高减半,这里没有做,而是分出去做了,直接就是分为两个分支了,所以最终宽高不变,维度也不变
#--------------------------------------------------------------------#
class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        #                                       Res次数, shortcut
        super().__init__()
        # 两个分支中间维度缩减一半
        hidden_channels = int(out_channels * expansion)

        #--------------------------------------------------#
        #   右侧: 主干部分的初次卷积 1x1Conv
        #--------------------------------------------------#
        self.conv1  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #--------------------------------------------------#
        #   根据循环的次数构建上述Bottleneck残差结构
        #--------------------------------------------------#
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.m      = nn.Sequential(*module_list)

        #--------------------------------------------------#
        #   左侧: 大的残差边部分的初次卷积(只有一次1x1Conv)
        #--------------------------------------------------#
        self.conv2  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)

        #-----------------------------------------------#
        #   最终: 对堆叠的结果进行卷积的处理 1x1Conv
        #-----------------------------------------------#
        self.conv3  = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x):
        #-------------------------------#
        #   x_1是主干部分
        #-------------------------------#
        x_1 = self.conv1(x)
        #-----------------------------------------------#
        #   主干部分利用残差结构堆叠继续进行特征提取
        #-----------------------------------------------#
        x_1 = self.m(x_1)

        #-------------------------------#
        #   x_2是大的残差边部分
        #-------------------------------#
        x_2 = self.conv2(x)

        #-----------------------------------------------#
        #   主干部分和大的残差边部分进行堆叠
        #-----------------------------------------------#
        x = torch.cat((x_1, x_2), dim=1)
        #-----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        #-----------------------------------------------#
        return self.conv3(x)

#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化,增大感受野
#   池化后和输入数据进行维度堆叠
#   pool_sizes=[1, 5, 9, 13] 1不变,所以不用做了
#---------------------------------------------------#
class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        # 1x1Conv缩减通道
        self.conv1      = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        #                                                             stride=1且有padding,所以最终大小不变
        self.m          = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels  = hidden_channels * (len(kernel_sizes) + 1)
        # 1x1Conv调整通道
        self.conv2      = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        # maxpool时kernel=1不用做,所以要加上[x]
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

class CSPDarknet(nn.Module):
    def __init__(self,
                dep_mul,    # 深度系数
                wid_mul,    # 宽度系数
                out_features=("dark3", "dark4", "dark5"),   # 返回的层数
                depthwise=False,    # 是否使用深度可分离卷积
                act="silu",):       # 激活函数
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        #-----------------------------------------------#
        base_channels   = int(wid_mul * 64)  # 64
        #   每个stage的
        base_depth      = max(round(dep_mul * 3), 1)  # 3

        #-----------------------------------------------#
        #   利用focus网络结构进行特征提取
        #           focus特征提取       卷积
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        #-----------------------------------------------#
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        #-----------------------------------------------#
        #   完成卷积之后，320, 320, 64 -> 160, 160, 128
        #   完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        #-----------------------------------------------#
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        )

        #-----------------------------------------------#
        #   完成卷积之后，160, 160, 128 -> 80, 80, 256
        #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        #-----------------------------------------------#
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        #-----------------------------------------------#
        #   完成卷积之后，80, 80, 256 -> 40, 40, 512
        #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        #-----------------------------------------------#
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        #-----------------------------------------------#
        #   完成卷积之后，40, 40, 512 -> 20, 20, 1024
        #   完成SPP之后，20, 20, 1024 -> 20, 20, 1024
        #   完成CSPlayer之后，20, 20, 1024 -> 20, 20, 1024
        #-----------------------------------------------#
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        #-----------------------------------------------#
        #   dark3的输出为80, 80, 256，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        outputs["dark3"] = x
        #-----------------------------------------------#
        #   dark4的输出为40, 40, 512，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        outputs["dark4"] = x
        #-----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


if __name__ == '__main__':
    print(CSPDarknet(1, 1))