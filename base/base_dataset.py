#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/25 14:34
# @Author : caius
# @Site : 
# @File : base_dataset.py
# @Software: PyCharm

import copy
from torch.utils.data import Dataset
# from data_loader.modules import *
import cv2
import numpy as np
from PIL import Image
from data_loader.modules import *




class BaseDataSet(Dataset):

    def __init__(self, data_path: str, img_mode,  transform=None,
                 target_transform=None ,crop_size=(320, 256), scale=True, mirror=True, ignore_label=255):
        assert img_mode in ['RGB', 'BRG', 'GRAY']

        # print('dataset', data_path)
        self.data_list = self.load_data(data_path[0])
        item_keys = ['img_path', 'img_mask', 'img_name']
        for item in item_keys:
            assert item in self.data_list[0], 'data_list from load_data must contains {}'.format(item_keys)
        self.img_mode = img_mode
        self.transform = transform
        self.target_transform = target_transform
        # self._init_pre_processes(pre_processes)
        self.is_mirror = mirror
        self.mean = (128, 128, 128)
        self.id_to_trainid = {1: 1, 2: 1, 3: 1, 4: 1, 11: 2, 12: 2, 13: 2, 14: 2}
        print('{} images are loaded!'.format(len(self.data_list)))
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label


    def _init_pre_processes(self, pre_processes):
        self.aug = []
        if pre_processes is not None:
            for aug in pre_processes:
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']
                if isinstance(args, dict):
                    cls = eval(aug['type'])(**args)
                else:
                    cls = eval(aug['type'])(args)
                self.aug.append(cls)

    def load_data(self, data_path: str) -> list:
        """
        把数据加载为一个list：
        :params data_path: 存储数据的文件夹或者文件
        return a dict ,包含了，'img_path','img_name','text_polys','texts','ignore_tags'
        """
        raise NotImplementedError

    def generate_scale_label(self, image, label):
        """

        """
        raise NotImplementedError

    def id2trainId(self, label, reverse=False):
        raise NotImplementedError

    def apply_pre_processes(self, data):
        for aug in self.aug:
            data = aug(data)
        return data

    def __getitem__(self, index):
        try:
            rdata = []
            data = copy.deepcopy(self.data_list[index])
            #  "img_path": image_path,
            #  "img_mask": label_path,
            #  "img_name": name
            im = cv2.imread(data['img_path'], 1 if self.img_mode != 'GRAY' else 0)
            if self.img_mode == 'RGB':
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # data = self.apply_pre_processes(data)
            label = Image.open(data["img_mask"])
            label = np.array(label)
            # print(np.unique(label))
            # print('label channel{}'.format(label.shape))
            # print('label scale{}'.format(np.unique(label)))
            label = self.id2trainId(label)
            # if self.transform:
            #     data['img'] = self.transform(data['img'])
            if self.scale:
                image, label = self.generate_scale_label(im, label)

            image = np.asarray(image, np.float32)
            # print('image', image.shape, 'label:', label.shape)
            img_h, img_w = label.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                             pad_w, cv2.BORDER_CONSTANT,
                                             value=(0.0, 0.0, 0.0))
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                               pad_w, cv2.BORDER_CONSTANT,
                                               value=(self.ignore_label,))
            else:
                img_pad, label_pad = image, label

            img_h, img_w = label_pad.shape
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
            # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
            image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
            label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

            # image = image.transpose((2, 0, 1))
            # print('iamge shape',image.shape)
            # if self.is_mirror:
            #     flip = np.random.choice(2) * 2 - 1
            #     image = image[:, :, ::flip]
            #     label = label[:, ::flip]
            dict2 = {'img': image,'label':label}
            data.update(dict2)
            if self.transform:
                data['img'] = self.transform(data['img'])
            # # print(image.shape, label.shape)
            # rdata['img'] = image
            # rdata['label'] = label
            # rdata['img_name'] = ['img_name']
            return data
        except:
            return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        return len(self.data_list)