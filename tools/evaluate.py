#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/7 22:47
# @Author : caius
# @Site : 
# @File : eval.py.py
# @Software: PyCharm
import os
import sys
import scipy.ndimage as ndimage
# from utils import runningScore
import numpy as np
from math import ceil
import torch.nn as nn
import torch.functional as F
from PIL import Image as PILImage
import torch.distributed as dist
project = 'ESCNN' # 工作项目录

sys.path.append(os.getcwd().split(project)[0] + project)
import argparse
import time
import torch
from tqdm.auto import   tqdm

class EVAL():
    def __init__(self, model_path,gpu_id=0):
        from models import get_model
        from data_loader import  get_dataloader
        #from utils import  get_metric
        self.model_path =model_path

        self.device = torch.device("cuda:%s" % gpu_id)
        if gpu_id is not None:
            torch.backends.cudnn.benchmark = True
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        # print(checkpoint['state_dict'])
        config = checkpoint['config']
        config['distributed'] =False
        config['arch']['args']['pretrained'] = False
        config['arch']['args']['training'] = False
        self.distributed = config['distributed']
        self.validate_loader = get_dataloader(config['dataset']['validate'],  self.distributed)

        self.model = get_model(config['arch'])
        self.model = nn.DataParallel(self.model)
        # print(self.model)
        # exit()
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
    def all_reduce_tensor(self, tensor, norm=True):
        if self.distributed:
            return self.all_reduce_tensor2(tensor, world_size=self.world_size, norm=norm)
        else:
            print('ooooooooooooooooook')
            return torch.mean(tensor)

    def all_reduce_tensor2(self,tensor, op=dist.ReduceOp.SUM, world_size=1, norm=True):
        tensor = tensor.clone()
        dist.all_reduce(tensor, op)
        if norm:
            tensor.div_(world_size)

        return tensor

    def pad_image(self,img, target_size):
        """Pad an image up to the target size."""
        rows_missing = target_size[0] - img.shape[2]
        cols_missing = target_size[1] - img.shape[3]
        padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
        return padded_img

    def get_palette(self,num_cls):
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

    def predict_sliding(self,net, image, tile_size, classes, recurrence):
        interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
        image_size = image.shape
        overlap = 1 / 3

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
                padded_img = self.pad_image(img, tile_size)
                # plt.imshow(padded_img)
                # plt.show()
                tile_counter += 1
                # print("Predicting tile %i" % tile_counter)
                padded_prediction = net(torch.from_numpy(padded_img).cuda(non_blocking=True))
                if isinstance(padded_prediction, list):
                    padded_prediction = padded_prediction[0]
                padded_prediction = interp(padded_prediction).cpu().numpy().transpose(0, 2, 3, 1)
                prediction = padded_prediction[0, 0:img.shape[2], 0:img.shape[3], :]
                count_predictions[0, y1:y2, x1:x2] += 1
                full_probs[:, y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

        # average the predictions in the overlapping regions
        full_probs /= count_predictions
        # visualize normalization Weights
        # plt.imshow(np.mean(count_predictions, axis=2))
        # plt.show()
        return full_probs

    def get_confusion_matrix(self,gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

    # self.model, batch['img'], (320, 256), [1.0], 3, False, 0)
    def predict_multiscale(self,net, image, tile_size, scales, classes, flip_evaluation, recurrence):
        """
        Predict an image by looking at it with different scales.
            We choose the "predict_whole_img" for the image with less than the original input size,
            for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
        """
        image = image.data.cpu()
        N_, C_, H_, W_ = image.shape
        full_probs = np.zeros((N_, H_, W_, classes))
        for scale in scales:
            scale = float(scale)
            scale_image = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
            # scaled_probs = predict_whole(net, scale_image, tile_size, recurrence)
            scaled_probs = self.predict_sliding(net, scale_image, tile_size, classes, recurrence)
            if flip_evaluation == True:
                # flip_scaled_probs = predict_whole(net, scale_image[:,:,:,::-1].copy(), tile_size, recurrence)
                flip_scaled_probs = self.predict_sliding(net, scale_image[:, :, :, ::-1].copy(), tile_size, classes,
                                                    recurrence)
                scaled_probs = 0.5 * (scaled_probs + flip_scaled_probs[:, ::-1, :])
            full_probs += scaled_probs
        full_probs /= len(scales)
        return full_probs

    def eval(self):
        self.model.eval()
        # torch.cuda.empty_cache()  # speed up evaluating after training finished
        total_frame = 0.0
        total_time = 0.0
        confusion_matrix = np.zeros((3, 3))
        palette = self.get_palette(256)
        save_path = os.path.join(os.path.dirname(self.model_path), 'outputs')
        save_gary_path = os.path.join(os.path.dirname(self.model_path), 'outputs_gray')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_gary_path):
            os.makedirs(save_gary_path)
        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            with torch.no_grad():
                # 数据进行转换和丢到gpu
                for key, value in batch.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                start = time.time()
                target = batch['label']
                h, w = target.size(1), target.size(2)
                output = self.predict_multiscale(self.model, batch['img'], (256,320), [1.0], 3, False, 0)
                seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
                seg_gt = np.asarray(target.cuda().data.cpu().numpy(), dtype=np.int)
                for i in range(batch['img'].size(0)):
                    output_im = PILImage.fromarray(seg_pred[i])
                    output_im.save(os.path.join(save_gary_path, batch['img_name'][i] + '.png'))
                    output_im.putpalette(palette)
                    output_im.save(os.path.join(save_path,  batch['img_name'][i] + '.png'))

                ignore_index = seg_gt != 255

                seg_gt = seg_gt[ignore_index]
                seg_pred = seg_pred[ignore_index]

                # show_all(gt, output)
                confusion_matrix += self.get_confusion_matrix(seg_gt, seg_pred, 3)


        confusion_matrix = torch.from_numpy(confusion_matrix).contiguous().cuda()
        confusion_matrix = self.all_reduce_tensor(confusion_matrix, norm=False).cpu().numpy()
        # print(confusion_matrix)
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()

        total_time += time.time() - start
        total_frame += batch['img'].size()[0]


        print('FPS:{}'.format(total_frame / total_time))
        return {'meanIU':mean_IU, 'IU_array':IU_array,'FPS':'{}'.format(total_frame / total_time)}

def init_args():
    parser = argparse.ArgumentParser(description='ESCNN')
    parser.add_argument('--model_path', required=True, type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    project = 'ESCNN2'  # 工作项目根目录
    # print(os.getcwd().split(project)[0] )
    sys.path.append(os.getcwd().split(project)[0])
    args = init_args()
    eval = EVAL(args.model_path)
    result = eval.eval()
    print(result)
