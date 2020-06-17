#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/1 14:52
# @Author : caius
# @Site : 
# @File : data_set_utils.py
# @Software: PyCharm
import os, glob
import random
/data/datasets/melons/dataset/training_set/bmp/1006-2/42.bmp	/data/datasets/melons/dataset/training_set/label/1006-2/0930_gt.png
/data/test/images/1123-1_16.bmp	/data/test/masks/1123-1_16.png
root = '/data/test/images'
f_w = open(os.path.join('./', 'test.txt'), 'w', encoding='utf8')
images = []

images += glob.glob(os.path.join(root,  '*.bmp'))
#     images += glob.glob(os.path.join(root, name, '*.jpg'))
#     images += glob.glob(os.path.join(root, name, '*.jpeg'))
random.shuffle(images)
for line in images:
    seg =line.replace('images', 'masks').replace('bmp', 'png')
    
    f_w.write("%s\t%s\n" % (line, seg))
    # f_w.write(line)
f_w.close()
print('Write Done!')