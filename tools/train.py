#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/25 18:26
# @Author : caius
# @Site : 
# @File : train.py.py
# @Software: PyCharm
from __future__ import print_function

import argparse
import os
# from utils import parse_config
import anyconfig





def init_args():
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--config_file', default='../config/melons_aspp.yaml', type=str)
    parser.add_argument('--local_rank', dest='local_rank', default=0, type=int, help='Use distributed training')

    args = parser.parse_args()
    return args


def main(config):
    import torch
    from models import get_model,get_criterion
    from data_loader import get_dataloader
    from trainer import Trainer
    # from utils import get_metric
    # print('local rank',args.local_rank)
    torch.autograd.set_detect_anomaly(True)
    if config['distributed']:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=torch.cuda.device_count(),
                                             rank=args.local_rank)

    config['local_rank'] = args.local_rank
    train_loader = get_dataloader(config['dataset']['train'], config['distributed'])
    print(config['distributed'])
    assert train_loader is not None
    if 'validate' in config['dataset']:
        validate_loader = get_dataloader(config['dataset']['validate'], config['distributed'])
    else:
        validate_loader = None
    criterion = get_criterion(config['loss']).cuda()

    model = get_model(config['arch'])
    if config['distributed'] :
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    trainer = Trainer(config=config,
                      model=model,
                      criterion=criterion,
                      train_loader=train_loader,
                      validate_loader=validate_loader)
    trainer.train()

if __name__ == '__main__':
    import sys

    project = 'ESCNN2'  # 工作项目根目录
    # print(os.getcwd().split(project)[0] )
    sys.path.append(os.getcwd().split(project)[0])
    from utils import parse_config


    args = init_args()
    print(args.config_file)
    assert os.path.exists(args.config_file)
    config = anyconfig.load(open(args.config_file, 'rb'))
    if 'base' in config:
        config = parse_config(config)
    main(config)