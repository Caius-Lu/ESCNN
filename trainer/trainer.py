#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/25 17:16
# @Author : caius
# @Site : 
# @File : trainer.py
# @Software: PyCharm

import time

import torch
import torchvision.utils as vutils
from tqdm import tqdm
from torch.nn import functional as F
import numpy as np
import torch.distributed as dist
from base import BaseTrainer
from utils import WarmupPolyLR, runningScore, decode_predictions, decode_labels


class Trainer(BaseTrainer):
    def __init__(self, config, model, criterion, train_loader, validate_loader, post_process=None):
        super(Trainer, self).__init__(config, model, criterion)
        self.show_images_iter = self.config['trainer']['show_images_iter']
        self.train_loader = train_loader
        self.validate_loader = validate_loader
        self.post_process = post_process
        self.train_loader_len = len(train_loader)
        if self.config['lr_scheduler']['type'] == 'WarmupPolyLR':
            warmup_iters = config['lr_scheduler']['args']['warmup_epoch'] * self.train_loader_len
            if self.start_epoch > 1:
                self.config['lr_scheduler']['args']['last_epoch'] = (self.start_epoch - 1) * self.train_loader_len
            self.scheduler = WarmupPolyLR(self.optimizer, max_iters=self.epochs * self.train_loader_len,
                                          warmup_iters=warmup_iters, **config['lr_scheduler']['args'])
        if self.validate_loader is not None:
            self.logger_info(
                'train dataset has {} samples,{} in dataloader, validate dataset has {} samples,{} in dataloader'.format(
                    len(self.train_loader.dataset), self.train_loader_len, len(self.validate_loader.dataset),
                    len(self.validate_loader)))
        else:
            self.logger_info('train dataset has {} samples,{} in dataloader'.format(len(self.train_loader.dataset),
                                                                                    self.train_loader_len))

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_start = time.time()
        batch_start = time.time()
        train_loss = 0.
        running_metric_melons = runningScore(3)
        lr = self.optimizer.param_groups[0]['lr']

        for i, batch in enumerate(self.train_loader):
            if i >= self.train_loader_len:
                break
            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']
            print(self.optimizer, self.config['local_rank'])

            # 数据进行转换和丢到gpu
            for key, value in batch.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)

            cur_batch_size = batch['img'].size()[0]
            # print('image name :',batch['img_name'])
            self.optimizer.zero_grad()
            preds = self.model(batch['img'])
            loss_dict = self.criterion(preds, batch)
            # backward
            if isinstance(preds, tuple):
                preds = preds[0]
            # print('preds:', preds.shape)

            # 反向传播时：在求导时开启侦测
            # print(loss_dict['loss'])
            # exit()
            reduce_loss = self.all_reduce_tensor(loss_dict['loss'])
            with torch.autograd.detect_anomaly():
                # loss.backward()
                loss_dict['loss'].backward()
            self.optimizer.step()
            if self.config['lr_scheduler']['type'] == 'WarmupPolyLR':
                self.scheduler.step()
            # acc iou
            target = batch['label']
            h, w = target.size(1), target.size(2)
            scale_pred = F.interpolate(input=preds, size=(h, w), mode='bilinear', align_corners=True)
            label_preds = torch.argmax(scale_pred, dim=1)
            running_metric_melons.update(target.data.cpu().numpy(), label_preds.data.cpu().numpy())
            score_, _ = running_metric_melons.get_scores()

            # loss 和 acc 记录到日志
            loss_str = 'loss: {:.4f}, '.format(reduce_loss.item())
            for idx, (key, value) in enumerate(loss_dict.items()):
                loss_dict[key] = value.item()
                if key == 'loss':
                    continue
                loss_str += '{}: {:.4f}'.format(key, loss_dict[key])
                if idx < len(loss_dict) - 1:
                    loss_str += ', '

            train_loss += loss_dict['loss']
            print(train_loss / self.train_loader_len, self.config['local_rank'])
            acc = score_['Mean Acc']
            iou_Mean_map = score_['Mean IoU']
            if self.global_step % self.log_iter == 0:
                batch_time = time.time() - batch_start
                self.logger_info(
                    '[{}/{}], [{}/{}], global_step: {}, speed: {:.1f} samples/sec, acc: {:.4f}, iou_Mean_map: {:.4f}, {}, lr:{:.6}, time:{:.2f}'.format(
                        epoch, self.epochs, i + 1, self.train_loader_len, self.global_step,
                                            self.log_iter * cur_batch_size / batch_time, acc,
                        iou_Mean_map, loss_str, lr, batch_time))
                batch_start = time.time()
            # print('loss_str', loss_str)

            if self.tensorboard_enable and self.config['local_rank'] == 0:
                # write tensorboard
                for key, value in loss_dict.items():
                    self.writer.add_scalar('TRAIN/LOSS/{}'.format(key), value, self.global_step)
                    self.writer.add_scalar('TRAIN/ACC_IOU/acc', acc, self.global_step)
                    self.writer.add_scalar('TRAIN/ACC_IOU/iou_Mean_map', iou_Mean_map, self.global_step)
                    self.writer.add_scalar('TRAIN/lr', lr, self.global_step)
                if self.global_step % self.show_images_iter == 0:
                    # show images on tensorboard
                    self.inverse_normalize(batch['img'])
                    preds_colors = decode_predictions(preds, cur_batch_size, 3)
                    self.writer.add_images('TRAIN/imgs', batch['img'][0].unsqueeze(0), self.global_step)
                    target = batch['label']
                    # (8, 256, 320, 3)

                    targets_colors = decode_labels(target, cur_batch_size, 3)
                    self.writer.add_image('TRAIN/labels', targets_colors[0], self.global_step, dataformats='HWC')
                    self.writer.add_image('TRAIN/preds', preds_colors[0], self.global_step, dataformats='HWC')
        return {'train_loss': train_loss / self.train_loader_len, 'lr': lr, 'time': time.time() - epoch_start,
                'epoch': epoch, 'MeanIoU': iou_Mean_map}
    def all_reduce_tensor(self, tensor2, norm=True):
        if self.distributed:
            return self.all_reduce_tensor2(tensor2, world_size=self.world_size, norm=norm)
        else:
            return torch.mean(tensor)

    def all_reduce_tensor2(self,tensor2, op=dist.ReduceOp.SUM, world_size=1, norm=True):
        tensor = tensor2.clone()
        dist.all_reduce(tensor2, op)
        if norm:
            tensor2.div_(world_size)

        return tensor2
    def _eval(self, epoch):
        self.model.eval()
        # torch.cuda.empty_cache()  # speed up evaluating after training finished
        total_frame = 0.0
        total_time = 0.0
        running_metric_melons = runningScore(3)
        mean_acc = []
        mean_iou = []
        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            with torch.no_grad():
                # 数据进行转换和丢到gpu
                for key, value in batch.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                start = time.time()
                # print(batch['img'].shape)
                # exit()
                preds = self.model(batch['img'])

                if isinstance(preds, tuple):
                    preds = preds[0]
                target = batch['label']
                h, w = target.size(1), target.size(2)
                scale_pred = F.interpolate(input=preds, size=(h, w), mode='bilinear', align_corners=True)
                label_preds = torch.argmax(scale_pred, dim=1)

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

    def _on_epoch_finish(self):
        self.logger_info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
            self.epoch_result['epoch'], self.epochs, self.epoch_result['train_loss'], self.epoch_result['time'],
            self.epoch_result['lr']))
        net_save_path = '{}/model_latest.pth'.format(self.checkpoint_dir)

        if self.config['local_rank'] == 0:
            save_best = False
            if self.validate_loader is not None:  # 使用meaniou作为最优模型指标
                acc, MeanIoU = self._eval(self.epoch_result['epoch'])

                if self.tensorboard_enable:
                    self.writer.add_scalar('EVAL/acc', acc, self.global_step)
                    self.writer.add_scalar('EVAL/MeanIoU', MeanIoU, self.global_step)
                self.logger_info('test: acc: {:.6f}, MeanIoU: {:.6f}'.format(acc, MeanIoU))

                if MeanIoU >= self.metrics['MeanIoU']:
                    save_best = True
                    self.metrics['train_loss'] = self.epoch_result['train_loss']
                    self.metrics['MeanIoU'] = MeanIoU
                    self.metrics['Mean Acc'] = acc
                    self.metrics['best_model_epoch'] = self.epoch_result['epoch']

            else:
                if self.epoch_result['MeanIoU'] <= self.metrics['MeanIoU']:
                    save_best = True
                    self.metrics['MeanIoU'] = self.epoch_result['MeanIoU']
                    self.metrics['best_model_epoch'] = self.epoch_result['epoch']
            best_str = 'current best, '
            for k, v in self.metrics.items():
                best_str += '{}: {:.6f}, '.format(k, v)
            self.logger_info(best_str)
            self._save_checkpoint(self.epoch_result['epoch'], net_save_path, save_best)
        

    def _on_train_finish(self):
        # for k, v in self.metrics.items():
        #     self.logger_info('{}:{}'.format(k, v))
        self.logger_info('finish train')
