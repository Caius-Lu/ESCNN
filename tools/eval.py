#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/7 22:47
# @Author : caius
# @Site : 
# @File : eval.py.py
# @Software: PyCharm
import os
from PIL import Image as PILImage
import sys

import numpy as np
from torch.nn import functional as F


project = 'ESCNN2' # 工作项目录

sys.path.append(os.getcwd().split(project)[0] + project)
import argparse
import time
import torch
from tqdm.auto import   tqdm

class EVAL():
    def __init__(self, model_path,gpu_id=0):
        from models import get_model

        from utils import runningScore
        from data_loader import  get_dataloader
        self.runningScore = runningScore
        #from utils import  get_metric
        self.device = torch.device("cuda:%s" % gpu_id)
        self.model_path = model_path
        if gpu_id is not None:
            torch.backends.cudnn.benchmark = True
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        config = checkpoint['config']
        config['arch']['args']['pretrained'] = False
        config['arch']['args']['training'] = False
        self.validate_loader = get_dataloader(config['dataset']['validate'], config['distributed'])

        self.model = get_model(config['arch'])
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
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
    def eval(self):
        self.model.eval()
        # torch.cuda.empty_cache()  # speed up evaluating after training finished
        total_frame = 0.0
        total_time = 0.0
        running_metric_melons = self.runningScore(3)
        mean_acc = []
        mean_iou = []
        palette = self.get_palette(256)
        save_path = os.path.join(os.path.dirname(self.model_path), 'outputs')
        save_gary_path = os.path.join(os.path.dirname(self.model_path), 'outputs_gray')
        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            with torch.no_grad():
                # 数据进行转换和丢到gpu
                for key, value in batch.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                start = time.time()
                preds = self.model(batch['img'])
                # print(preds.shape)
                if isinstance(preds, tuple):
                    preds = preds[0]
                target = batch['label']
                h, w = target.size(1), target.size(2)
                scale_pred = F.interpolate(input=preds, size=(h, w), mode='bilinear', align_corners=True)
                label_preds = torch.argmax(scale_pred, dim=1)
                # seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
                seg_pred = np.asarray(label_preds.cuda().data.cpu().numpy(), dtype=np.int)
                print(type(seg_pred))
                print(seg_pred.shape)
                print(seg_pred[0].shape)
                # exit()
                for i in range(batch['img'].size(0)):
                    output_im = PILImage.fromarray(seg_pred[i].astype('uint8'))
                    output_im.save(os.path.join(save_gary_path, batch['img_name'][i] + '.png'))
                    output_im.putpalette(palette)
                    output_im.save(os.path.join(save_path, batch['img_name'][i] + '.png'))
                running_metric_melons.update(target.data.cpu().numpy(), label_preds.data.cpu().numpy())
                score_, _ = running_metric_melons.get_scores()
                total_time += time.time() - start
                total_frame += batch['img'].size()[0]
                acc = score_['Mean Acc']
                iou_Mean_map = score_['Mean IoU']
                mean_acc.append(acc)
                mean_iou.append(iou_Mean_map)

        print('FPS:{}'.format(total_frame / total_time))
        return np.array(mean_acc).mean(), np.array(mean_iou).mean()

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
