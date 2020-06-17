#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/25 15:32
# @Author : caius
# @Site :
# @File : segmentation_body.py
# @Software: PyCharm
import torch.nn as nn
import torch
from torch.nn import functional as F
from .attention import NonLocalBlock
import utils.mynn as mynn

class ConvHead(nn.Module):
    def __init__(self, in_channels=1024, out_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        return self.conv(x)



# class ConvHead(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.conv(x)


class ESHead(nn.Module):
    def __init__(self, in_channels=2048, num_classes=3):
        super().__init__()
        self.head = self.head = nn.Sequential(PSPModule(2048, 512),
                                                    nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.dsn = ConvHead()

        # self.weights_init(in_channels)

    def forward(self, x,x2):
        # torch.Size([2, 2048, 80, 64])
        # [2, 1024, 81, 65
        # print(x.shape)
        # exit()
        x = self.head(x)
        x_dsn = self.dsn(x2)
        return x, x_dsn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PSPModule(nn.Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = nn.Sequential(
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

# class PSPModule(nn.Module):
#     """
#     Reference:
#         Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
#     """

#     def __init__(self, in_features=2048, out_features=512, sizes=(1, 2, 3, 6)):
#         super(PSPModule, self).__init__()

#         self.stages = []
#         self.stages = nn.ModuleList([self._make_stage(in_features, out_features, size) for size in sizes])
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(in_features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1,
#                       bias=False),
#             nn.BatchNorm2d(out_features),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(0.3)
#         )
#         self.spa = SpatialAttention(kernel_size=3)
#         self._gate_conv = nn.Sequential(
#             nn.BatchNorm2d(out_features + 1),
#             nn.Conv2d(out_features + 1, out_features + 1, 1),
#             nn.ReLU(),
#             nn.Conv2d(out_features + 1, 1, 1),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#         self.attention = NonLocalBlock(channel=512)

#     def _make_stage(self, features, out_features, size):
#         prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
#         conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
#         bn = nn.Sequential(
#             nn.BatchNorm2d(out_features),
#             nn.ReLU(inplace=True)
#         )
#         return nn.Sequential(prior, conv, bn)

#     def forward(self, feats):
#         h, w = feats.size(2), feats.size(3)
#         priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
#                   self.stages] + [feats]
#         bottle = self.bottleneck(torch.cat(priors, 1))
#         # print('bottle:  shape', bottle.shape)
#         # bottle2 = self.attention(bottle)
#         # gating_features = self.spa(bottle2)
#         # alphas = self._gate_conv(torch.cat([bottle, gating_features], dim=1))
#         # input_features = (bottle * (alphas + 1))
#         return bottle