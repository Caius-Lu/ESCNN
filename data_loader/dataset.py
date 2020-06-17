#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/25 18:41
# @Author : caius
# @Site : 
# @File : dataset.py
# @Software: PyCharm
import pathlib
import os
import cv2
import numpy as np
import scipy.io as sio
import random
from tqdm.auto import tqdm
from torchvision import transforms

from base import  BaseDataSet
from utils import get_datalist


class melonDataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode,eval_mode, transform=None,
                 crop_size=(320, 256), ignore_label=255):
        super().__init__(data_path, img_mode,eval_mode,transform,crop_size,ignore_label)


    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy


    def load_data(self, data_path: str) -> list:
        data_list = get_datalist(data_path)
        # t_data_list = []
        # for img_path, label_path in data_list:
        #     data = self._get_annotation(label_path)
        #     if len(data['text_polys']) > 0:
        #         item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
        #         item.update(data)
        #         t_data_list.append(item)
        #     else:
        #         print('there is no suit bbox in {}'.format(label_path))
        return data_list


def get_transforms(transforms_config):
    tr_list = []
    for item in transforms_config:
        if 'args' not in item:
            args = {}
        else:
            args = item['args']
        cls = getattr(transforms, item['type'])(**args)
        tr_list.append(cls)
    tr_list = transforms.Compose(tr_list)
    return tr_list


if __name__ == '__main__':
    import torch
    import anyconfig
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from utils import parse_config, show_img, plt

    config = anyconfig.load('../config/melons_aspp.yaml')
    config = parse_config(config)
    dataset_args = config['dataset']['validate']['dataset']['args']
    # dataset_args.pop('data_path')
    if 'transforms' in dataset_args:
        img_transfroms = get_transforms(dataset_args.pop('transforms'))
    #data_list = [(r'/data/datasets/melons/dataset/training_set/bmp/1126-1/21.bmp	/data/datasets/melons/dataset/training_set/label/1126-1/1125_gt.png')]
    train_data = melonDataset(data_path=dataset_args.pop('data_path'),transform=img_transfroms,
                                  **dataset_args)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers=0)
    for i, data in enumerate(tqdm(train_loader)):
        image, mask,_ = data['img'],data['label'],data ['img_name']

        # rdata['img'] = image
        # rdata['label'] = label
        # rdata['img_name'] = ['img_name']

        print(image.shape, mask.shape)
        # show_img(image[0].numpy().transpose(1, 2, 0), title='img')
        # show_img(label, title='label')
        # plt.show()
        pass
