#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/25 15:21
# @Author : caius
# @Site : 
# @File : __init__.py.py
# @Software: PyCharm

from . import model, loss, criterion


def get_model(config):
    _model = getattr(model, config['type'])(config['args'])
    return _model


def get_loss(config):
    return getattr(loss, config['type'])(**config['args'])

def get_criterion(config):
    return getattr(criterion, config['type'])(**config['args'])