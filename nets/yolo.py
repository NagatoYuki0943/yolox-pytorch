#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.nn as nn

from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv

#---------------------------------------------------#
#   分类和框的预测
#---------------------------------------------------#
class YOLOXHead(nn.Module):
    def __init__(self,
                num_classes,    # 分类数
                width = 1.0,    # 宽度
                in_channels = [256, 512, 1024], # 输入通道数
                act = "silu",       # 激活函数
                depthwise = False,):# dw卷积
        super().__init__()
        #-------------------------------------------#
        #   调整 channels 宽度
        #-------------------------------------------#
        in_channels = [int(i * width) for i in in_channels]
        hidden_channels = int(256 * width)

        Conv            = DWConv if depthwise else BaseConv

        self.stems      = nn.ModuleList()   # 第1次Conv+BN+ReLU
        self.cls_convs  = nn.ModuleList()   # 左侧第1次两个Conv+BN+ReLU
        self.cls_preds  = nn.ModuleList()   # 左侧种类预测
        self.reg_convs  = nn.ModuleList()   # 右侧第1次两个Conv+BN+ReLU
        self.reg_preds  = nn.ModuleList()   # 右侧特征点回归预测(中心位置,宽高)
        self.obj_preds  = nn.ModuleList()   # 右侧是否有物体预测

        for i in range(len(in_channels)):
            #---------------------------------------------------#
            #   利用1x1Conv进行通道整合
            #---------------------------------------------------#
            self.stems.append(BaseConv(in_channels[i], hidden_channels, ksize = 1, stride = 1, act = act))

            """左侧部分"""
            #---------------------------------------------------#
            #   利用两个3x3Conv+标准化+激活函数来进行特征提取
            #---------------------------------------------------#
            self.cls_convs.append(nn.Sequential(*[
                Conv(hidden_channels, hidden_channels, ksize = 3, stride = 1, act = act),
                Conv(hidden_channels, hidden_channels, ksize = 3, stride = 1, act = act),
            ]))
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            self.cls_preds.append(
                nn.Conv2d(hidden_channels, num_classes, kernel_size = 1, stride = 1, padding = 0)
            )

            """右侧部分"""
            #---------------------------------------------------#
            #   利用两个3x3Conv+标准化+激活函数来进行特征提取
            #---------------------------------------------------#
            self.reg_convs.append(nn.Sequential(*[
                Conv(hidden_channels, hidden_channels, ksize = 3, stride = 1, act = act),
                Conv(hidden_channels, hidden_channels, ksize = 3, stride = 1, act = act)
            ]))
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            self.reg_preds.append(
                nn.Conv2d(hidden_channels, 4, kernel_size = 1, stride = 1, padding = 0)
            )
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            self.obj_preds.append(
                nn.Conv2d(hidden_channels, 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs: (P3_out, P4_out, P5_out) 元组
        #       P3_out: 80, 80, 256
        #       P4_out: 40, 40, 512
        #       P5_out: 20, 20, 1024
        #---------------------------------------------------#

        outputs = []
        # 不同输入大小分别的判断
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1Conv进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)

            """左侧部分"""
            #---------------------------------------------------#
            #   利用两个3x3Conv+标准化+激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            """右侧部分"""
            #---------------------------------------------------#
            #   利用两个3x3Conv+标准化+激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            #---------------------------------------------------#
            #   通道上拼接 特征点,是否有物体,分类
            #   80, 80, 4+1+num_classes
            #   40, 40, 4+1+num_classes
            #   20, 20, 4+1+num_classes
            #---------------------------------------------------#
            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs

#---------------------------------------------------#
#   FPN结构
#---------------------------------------------------#
class YOLOPAFPN(nn.Module):
    def __init__(self,
                depth = 1.0,    # 深度
                width = 1.0,    # 宽度
                in_features = ("dark3", "dark4", "dark5"),  # CSPDarknet的返回层
                in_channels = [256, 512, 1024],             # CSPDarknet返回三层的channel
                depthwise = False,  # 是否使用dw卷积
                act = "silu"):      # 激活函数
        super().__init__()
        #-------------------------------------------#
        #   调整 in_channels 宽度
        #-------------------------------------------#
        in_channels = [int(i * width) for i in in_channels]

        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features

        #-------------------------------------------#
        #   上采样,宽高x2
        #-------------------------------------------#
        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   P5开始的的1x1Conv,得到 P5
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(in_channels[2], in_channels[1], 1, 1, act=act)

        #-------------------------------------------#
        #   P5_upsample,feat2 拼接之后的 CSPLayer
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            in_channels  = 2 * in_channels[1],
            out_channels = in_channels[1],
            n            = round(3 * depth),
            shortcut     = False,
            depthwise    = depthwise,
            act          = act,
        )

        #-------------------------------------------#
        #   拼接 P5_upsample,feat2 后的CSPLayer的1x1Conv,得到P4
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(in_channels[1], in_channels[0], 1, 1, act=act)
        #-------------------------------------------#
        #   拼接 P4_upsample,feat1 后的 CPSLayer,得到 P3_out
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            in_channels  = 2 * in_channels[0],
            out_channels = in_channels[0],
            n            = round(3 * depth),
            shortcut     = False,
            depthwise    = depthwise,
            act          = act,
        )

        #-------------------------------------------#
        #   P3_out下采样
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        self.bu_conv2       = Conv(in_channels[0], in_channels[0], 3, 2, act=act)
        #-------------------------------------------#
        #   拼接 P3_downsample, P4 后的 CPSLayer,得到 P4_out
        #   40, 40, 512 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_n3 = CSPLayer(
            in_channels  = 2 * in_channels[0],
            out_channels = in_channels[1],
            n            = round(3 * depth),
            shortcut     = False,
            depthwise    = depthwise,
            act          = act,
        )

        #-------------------------------------------#
        #   P4_out下采样
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        self.bu_conv1       = Conv(in_channels[1], in_channels[1], 3, 2, act=act)
        #-------------------------------------------#
        #   拼接 P4_downsample, P5 后的 CPSLayer,得到 P5_out
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        self.C3_n4 = CSPLayer(
            in_channels  = 2 * in_channels[1],
            out_channels = in_channels[2],
            n            = round(3 * depth),
            shortcut     = False,
            depthwise = depthwise,
            act = act,
        )

    def forward(self, input):
        # 返回的是字典,要取出来
        out_features            = self.backbone.forward(input)
        #-------------------------------------------#
        #   feat1: 80, 80, 256
        #   feat2: 40, 40, 512
        #   feat3: 20, 20, 1024
        #-------------------------------------------#
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]

        """第一次上采样"""
        #-------------------------------------------#
        #   P5开始的的1x1Conv,得到 P5
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.lateral_conv0(feat3)
        #-------------------------------------------#
        #   P5上采样
        #   20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #   拼接 P5_upsample,feat2
        #   40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #-------------------------------------------#
        #   P5_upsample,feat2 拼接之后的CSPLayer
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        """第二次上采样"""
        #-------------------------------------------#
        #   拼接 P5_upsample,feat2 后的CSPLayer的1x1Conv,得到P4
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample)
        #-------------------------------------------#
        #   P4的上采样
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4)
        #-------------------------------------------#
        #   拼接 P4_upsample,feat1
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1)
        #-------------------------------------------#
        #   拼接 P4_upsample,feat1 后的 CPSLayer,得到 P3_out
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out      = self.C3_p3(P4_upsample)

        """第一次下采样"""
        #-------------------------------------------#
        #   P3_out下采样
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        P3_downsample   = self.bu_conv2(P3_out)
        #-------------------------------------------#
        #   拼接 P3_downsample, P4
        #   40, 40, 256 + 40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P3_downsample   = torch.cat([P3_downsample, P4], 1)
        #-------------------------------------------#
        #   拼接 P3_downsample, P4 后的 CPSLayer,得到 P4_out
        #   40, 40, 512 -> 40, 40, 512
        #-------------------------------------------#
        P4_out          = self.C3_n3(P3_downsample)

        """第二次下采样"""
        #-------------------------------------------#
        #   P4_out下采样
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        P4_downsample   = self.bu_conv1(P4_out)
        #-------------------------------------------#
        #   拼接 P4_downsample, P5
        #   20, 20, 512 + 20, 20, 512 -> 20, 20, 1024
        #-------------------------------------------#
        P4_downsample   = torch.cat([P4_downsample, P5], 1)
        #-------------------------------------------#
        #   拼接 P4_downsample, P5 后的 CPSLayer,得到 P5_out
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        P5_out          = self.C3_n4(P4_downsample)

        #-------------------------------------------#
        #   P3_out: 80, 80, 256
        #   P4_out: 40, 40, 512
        #   P5_out: 20, 20, 1024
        #-------------------------------------------#
        return (P3_out, P4_out, P5_out)

class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        depth, width    = depth_dict[phi], width_dict[phi]
        depthwise       = True if phi == 'nano' else False

        self.backbone   = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.head       = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x):
        #-------------------------------------------#
        #   P3_out: 80, 80, 256
        #   P4_out: 40, 40, 512
        #   P5_out: 20, 20, 1024
        #-------------------------------------------#
        fpn_outs    = self.backbone.forward(x)

        #---------------------------------------------------#
        #   80, 80, 4+1+num_classes
        #   40, 40, 4+1+num_classes
        #   20, 20, 4+1+num_classes
        #---------------------------------------------------#
        outputs     = self.head.forward(fpn_outs)
        return outputs


if __name__ == "__main__":
    model = YoloBody(80, 's')
    state_dict = torch.load("F:\BaiduYunDownload\yolox_s.pth")
    model.load_state_dict(state_dict)