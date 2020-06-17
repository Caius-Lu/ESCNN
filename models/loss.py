
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/25 15:57
# @Author : caius
# @Site : 
# @File : los.py
# @Software: PyCharm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd


class OhemCrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255, thresh=0.7, min_kept=13654, factor=8): # 100000  label 255
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        # self.min_kept_ratio = float(min_kept_ratio)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    # 寻找阈值
    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor # 8
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0/factor, 1.0/factor), order=1)  # 双线性插值 shape (1, 19, 96, 96)
        target = nd.zoom(np_target, (1.0, 1.0/factor, 1.0/factor), order=0)  #  #最近邻插值  shape(1, 96, 96)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (factor*factor) #int(self.min_kept_ratio * n * h * w)   #100000/64 = 1562
        # print('min_keep:', min_kept)
        input_label = target.ravel().astype(np.int32)  #将多维阵列化为一维 shape(9216, )
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))  #軸1滾動到軸0、shape(19, 9216)  这里的19是类别数目

        valid_flag = input_label != self.ignore_label #label中有效位置(9216, )
        valid_inds = np.where(valid_flag)[0]  #(9013, )
        label = input_label[valid_flag] #有效label(9013, )

        num_valid = valid_flag.sum() #9013
        if min_kept >= num_valid:  #9013 > 0
            threshold = 1.0
        elif num_valid > 0:   #9013 > 0
            prob = input_prob[:,valid_flag]  #找出有效区域对应的prob
            pred = prob[label, np.arange(len(label), dtype=np.int32)]  #???    shape(9013, )
            threshold = self.thresh  # 0.7
            if min_kept > 0:   #1562>0
                k_th = min(len(pred), min_kept)-1   # min(9013, 1562)-1 = 1561
                new_array = np.partition(pred, k_th)   #排序并分为两个区，小于第1561個及大于第1561個  np.partition的工作流程可以看做是先对数组排序（升序
                new_threshold = new_array[k_th]     #第1561對应的pred 0.03323581
                if new_threshold > self.thresh:     # #返回的閾值只能≥0.7
                    threshold = new_threshold
        return threshold


    def generate_new_target(self, predict, target):


        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))  # 调换维度

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:,valid_flag]
            # print('prob:',prob)
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            # print('pred : {0},   threshold: {1}'.format(pred,threshold))
            a = np.isnan(input_prob).sum()
            # if a >=1:
            #     print('data contains nan')
            #     exit()
            kept_flag = pred <= threshold  #二次篩選：在255中找出pred≤0.7的位置
            valid_inds = valid_inds[kept_flag]     # nan len(valid_inds) 为0
            # print('Labels: {} {}'.format(len(valid_inds), threshold))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = torch.from_numpy(input_label.reshape(target.size())).long().cuda(target.get_device())

        return new_target


    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad

        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)
        return self.criterion(predict, target)
