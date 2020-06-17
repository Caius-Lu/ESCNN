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
import os
import transforms.joint_transforms as joint_transforms
import torchvision.transforms as transforms
import transforms.transforms as extended_transforms
from data_loader.modules import *

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


class BaseDataSet(Dataset):

    def __init__(self, data_path: str, img_mode, eval_mode,  transform=None, crop_size=(256,320), ignore_label=255):
        assert img_mode in ['RGB', 'BRG', 'GRAY']

        # print('dataset', data_path)
        self.data_list = self.load_data(data_path[0])
        item_keys = ['img_path', 'img_mask', 'img_name']
        for item in item_keys:
            assert item in self.data_list[0], 'data_list from load_data must contains {}'.format(item_keys)
        self.img_mode = img_mode
        self.transform = transform
        self.crop_size =(256,320)
        self.pre_size  = None
        self.eval_mode = eval_mode
        self.i = 0
        self.scale_min = 0.85
        self.scale_max = 1.5
        self.color_aug = 0.0001
        self.rotate =0.5
        self.bblur =False
        self.dump_images = False
        self.eval_scales = None
        eval_scales=None
        eval_flip = False
        self.eval_flip = eval_flip
        if eval_scales != None:
            self.eval_scales = [float(scale) for scale in eval_scales.split(",")]
        else:
            self.eval_scales = [1]

        self.id_to_trainid = {1: 1, 2: 1, 3: 1, 4: 1, 11: 2, 12: 2, 13: 2, 14: 2}
        print('{} images are loaded!'.format(len(self.data_list)))
        self.ignore_label = ignore_label
        self.mean_std = ([0.5081455, 0.5081455, 0.5081455], [0.25244877, 0.25244877, 0.25244877])



    def load_data(self, data_path: str) -> list:
        """
        把数据加载为一个list：
        :params data_path: 存储数据的文件夹或者文件
        return a dict ,包含了，'img_path','img_name','text_polys','texts','ignore_tags'
        """
        raise NotImplementedError


    def id2trainId(self, label, reverse=False):
        raise NotImplementedError

    def apply_pre_processes(self, data):
        for aug in self.aug:
            data = aug(data)
        return data

    def _eval_get_item(self, img, mask, scales, flip_bool,data):
        # return_imgs = []
        # for flip in range(int(flip_bool)+1):
        #     imgs = []
        #     if flip :
        #         img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #     for scale in scales:
        #         w,h = img.size
        #         target_w, target_h = int(w * scale), int(h * scale)
        #         resize_img =img.resize((target_w, target_h))
        #         tensor_img = transforms.ToTensor()(resize_img)
        #         final_tensor = transforms.Normalize(*self.mean_std)(tensor_img)
        #         imgs.append(final_tensor)
        #     return_imgs.append(imgs)
        dict2 = {'img': img, 'label': mask}
        data.update(dict2)
        if self.transform:
            data['img'] = self.transform(data['img'])
        return data

    def __getitem__(self, index):
        try:

            data = copy.deepcopy(self.data_list[index])
            # if self.eval_mode:
            #     print(data["img_mask"])
            # im = cv2.imread(data['img_path'], 1 if self.img_mode != 'GRAY' else 0)
            # if self.img_mode == 'RGB':
            #     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            label = Image.open(data["img_mask"])
            label = np.array(label)

            label = self.id2trainId(label)
            mask = Image.fromarray(label.astype(np.uint8))
            im = Image.open(data['img_path']).convert('RGB')
            if self.eval_mode:
                return self._eval_get_item(im, label, self.eval_scales, self.eval_flip,data)


            # Geometric image transformations
            train_joint_transform_list = [
                joint_transforms.RandomSizeAndCrop(self.crop_size,
                                                   False,
                                                   pre_size=self.pre_size,
                                                   scale_min=self.scale_min,
                                                   scale_max=self.scale_max,
                                                   ignore_index=self.ignore_label),
                joint_transforms.Resize(self.crop_size),
                joint_transforms.RandomHorizontallyFlip()]

            if self.rotate:
                train_joint_transform_list += [joint_transforms.RandomRotate(self.rotate)]

            train_joint_transform = joint_transforms.Compose(train_joint_transform_list)

            ## Image appearance transformations
            train_input_transform = []
            if self.color_aug:
                train_input_transform += [extended_transforms.ColorJitter(
                    brightness=self.color_aug,
                    contrast=self.color_aug,
                    saturation=self.color_aug,
                    hue=self.color_aug)]

            if self.bblur:
                train_input_transform += [extended_transforms.RandomBilateralBlur()]
            elif self.bblur:
                train_input_transform += [extended_transforms.RandomGaussianBlur()]
            else:
                pass
            train_input_transform = transforms.Compose(train_input_transform)
            target_transform = extended_transforms.MaskToTensor()

            target_train_transform = extended_transforms.MaskToTensor()

            # Image Transformations
            # if train_joint_transform is not None:  # train_joint_transform
            img, mask = train_joint_transform(im, mask)
            # if train_input_transform is not None:  # train_input_transform
            img = train_input_transform(img)
            # if target_train_transform is not None:
            mask = target_train_transform(mask)

            if self.dump_images:
                outdir = '/data/SSSSdump_imgs_/'
                os.makedirs(outdir, exist_ok=True)
                out_img_fn = os.path.join(outdir, '{}.png'.format(self.i))
                out_msk_fn = os.path.join(outdir, '{}s_mask.png'.format(self.i))
                print(out_img_fn)
                self.i+=1
                mask_img = colorize_mask(np.array(mask))
                img.save(out_img_fn)
                mask_img.save(out_msk_fn)
            dict2 = {'img': img,'label':mask}
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