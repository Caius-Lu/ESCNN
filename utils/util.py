#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/25 14:39
# @Author : caius
# @Site : 
# @File : utils.py
# @Software: PyCharm

import json
import pathlib
import time
import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from torch import  nn
import scipy.ndimage as nd
import torch
from collections import OrderedDict, defaultdict
from PIL import Image
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

def setup_logger(log_file_path:str = None):
    import logging
    logging._warn_preinit_stderr = 0
    logger = logging.getLogger('ESCNN')
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    logger.setLevel(level=logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file_path is not None:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    return logger


def parse_config(config: dict) -> dict:
    import anyconfig
    base_file_list = config.pop('base')
    base_config = {}
    for base_file in base_file_list:
        tmp_config = anyconfig.load(open(base_file, 'rb'))
        if 'base' in tmp_config:
            tmp_config = parse_config(tmp_config)
        anyconfig.merge(tmp_config, base_config)
        base_config = tmp_config
    anyconfig.merge(base_config, config)
    return base_config


def get_datalist(data_list_path):
    """
    获取训练和验证的数据list
    :param train_data_path: 训练的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :return:
    """
    train_data = []
    img_ids = [i_id.strip().split() for i_id in open(data_list_path)]
    # print('img_id',img_ids)
    for item in  img_ids:
        image_path, label_path = item
        # test——file /data/melons/dataset/data/new_image/1019-2_59.png
        name = image_path.split('.')[0]
        name = name.split('/')[-2] + '_' + name.split('/')[-1]
        # name =  name.split('/')[-1]
        # img_file = osp.join(self.root, image_path)
        # label_file = osp.join(self.root, label_path)
        # if image_path.exists() and image_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
        # item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}  'img_path', 'img_mask', 'img_name'
        train_data.append({
            "img_path": image_path,
            "img_mask": label_path,
            "img_name": name
        })
    return train_data



# def get_datalist(train_data_path):
#     """
#     获取训练和验证的数据list
#     :param train_data_path: 训练的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
#     :return:
#     """
#     train_data = []
#     for p in train_data_path:
#         with open(p, 'r', encoding='utf-8') as f:
#             for line in f.readlines():
#                 line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
#                 if len(line) > 1:
#                     img_path = pathlib.Path(line[0].strip(' '))
#                     label_path = pathlib.Path(line[1].strip(' '))
#                     if img_path.exists() and img_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
#                         train_data.append((str(img_path), str(label_path)))
#     return train_data


def show_img(imgs: np.ndarray, title='img'):
    color = (len(imgs.shape) == 3 and imgs.shape[-1] == 3)
    imgs = np.expand_dims(imgs, axis=0)
    for i, img in enumerate(imgs):
        plt.figure()
        plt.title('{}_{}'.format(title, i))
        plt.imshow(img, cmap=None if color else 'gray')
    plt.show()


def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    mask = mask.data.cpu().numpy()
    n, h, w = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs
def decode_predictions(preds, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    if isinstance(preds, list):
        preds_list = []
        for pred in preds:
            preds_list.append(pred[-1].data.cpu().numpy())
        preds = np.concatenate(preds_list, axis=0)
    else:
        preds = preds.data.cpu().numpy()

    preds = np.argmax(preds, axis=1)
    n, h, w = preds.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(preds[i, 0]), len(preds[i])))
        pixels = img.load()
        for j_, j in enumerate(preds[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs




def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img

def predict_sliding(net, image, tile_size, classes, recurrence):
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    image_size = image.shape
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    # print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((image_size[0], image_size[2], image_size[3], classes))
    count_predictions = np.zeros((1, image_size[2], image_size[3], classes))
    tile_counter = 0

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], image_size[3])
            y2 = min(y1 + tile_size[0], image_size[2])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = image[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            # plt.imshow(padded_img)
            # plt.show()
            tile_counter += 1
            # print("Predicting tile %i" % tile_counter)
            padded_prediction = net(torch.from_numpy(padded_img).cuda(non_blocking=True))
            if isinstance(padded_prediction, list):
                padded_prediction = padded_prediction[0]
            padded_prediction = interp(padded_prediction).cpu().numpy().transpose(0,2,3,1)
            prediction = padded_prediction[0, 0:img.shape[2], 0:img.shape[3], :]
            count_predictions[0, y1:y2, x1:x2] += 1
            full_probs[:, y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    # visualize normalization Weights
    # plt.imshow(np.mean(count_predictions, axis=2))
    # plt.show()
    return full_probs

def predict_whole(net, image, tile_size, recurrence):
    N_, C_, H_, W_ = image.shape
    image = torch.from_numpy(image)
    interp = nn.Upsample(size=(H_, W_), mode='bilinear', align_corners=True)
    prediction = net(image.cuda())
    if isinstance(prediction, list):
        prediction = prediction[0]
    prediction = interp(prediction).cpu().numpy().transpose(0,2,3,1)
    return prediction

def predict_multiscale(net, image, tile_size, scales, classes, flip_evaluation, recurrence):
    """
    Predict an image by looking at it with different scales.
        We choose the "predict_whole_img" for the image with less than the original input size,
        for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    image = image.data
    N_, C_, H_, W_ = image.shape
    full_probs = np.zeros((N_, H_, W_, classes))
    for scale in scales:
        scale = float(scale)
        scale_image = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
        # scaled_probs = predict_whole(net, scale_image, tile_size, recurrence)
        scaled_probs = predict_sliding(net, scale_image, tile_size, classes, recurrence)
        if flip_evaluation == True:
            # flip_scaled_probs = predict_whole(net, scale_image[:,:,:,::-1].copy(), tile_size, recurrence)
            flip_scaled_probs = predict_sliding(net, scale_image[:,:,:,::-1].copy(), tile_size, classes, recurrence)
            scaled_probs = 0.5 * (scaled_probs + flip_scaled_probs[:,::-1,:])
        full_probs += scaled_probs
    full_probs /= len(scales)
    return full_probs




def load_model(model, model_file, is_restore=False):
    import logging
    logging._warn_preinit_stderr = 0
    logger = logging.getLogger('ESCNN')
    t_start = time.time()
    if isinstance(model_file, str):
        device = torch.device('cpu')
        state_dict = torch.load(model_file, map_location=device)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        logger.warning('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model
