#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/25 15:23
# @Author : caius
# @Site : 
# @File : model.py
# @Software: PyCharm
import torch.nn.functional as F

from models.modules import *
from models.resnest import  *
from torchsummary import summary
from models.psp_resnet import *
from torch.nn import init


backbone_dict = {
    'resnest101': {'models': resnest101,'out': [256, 512, 1024, 2048]},
    'resnest101_aspp':{'models':resnest101_aspp,'out':[256, 512,1024,2048]},
    'resnet18': {'models': resnet18, 'out': [64, 128, 256, 512]},
    'deformable_resnet18': {'models': deformable_resnet18, 'out': [64, 128, 256, 512]},
    'resnet34': {'models': resnet34, 'out': [64, 128, 256, 512]},
    'resnet50': {'models': resnet50, 'out': [256, 512, 1024, 2048]},
    'deformable_resnet50': {'models': deformable_resnet50, 'out': [256, 512, 1024, 2048]},
    'resnet101': {'models': resnet101, 'out': [256, 512, 1024, 2048]},
    'resnet152': {'models': resnet152, 'out': [256, 512, 1024, 2048]},
    'shufflenetv2': {'models': shufflenet_v2_x1_0, 'out': [24, 116, 232, 464],
    'resnet101': {'models': resnet101, 'out': [256, 512, 1024, 2048]}

}

}
segmentation_body_dict = {'FPN': FPN, 'FPEM_FFM': FPEM_FFM}
segmentation_head_dict = {'conv': ConvHead, 'espp': ESHead}


class ESCNNModel(nn.Module):
    def __init__(self, model_config: dict):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_body = model_config['segmentation_body']['type']
        segmentation_head = model_config['segmentation_head']['type']

        assert backbone in backbone_dict, 'backbone must in: {}'.format(backbone_dict)
        assert segmentation_body in segmentation_body, 'segmentation_head must in: {}'.format(segmentation_body)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(segmentation_head_dict)
        self.training = model_config['training']
        backbone_model, backbone_out = backbone_dict[backbone]['models'], backbone_dict[backbone]['out']
        self.backbone = backbone_model(pretrained=pretrained)
        self.segmentation_body = segmentation_body_dict[segmentation_body](backbone_out,
                                                                           **model_config['segmentation_body']['args'])
        self.segmentation_head = segmentation_head_dict[segmentation_head](self.segmentation_body.out_channels,
                                                                           **model_config['segmentation_head']['args'])
       
        self.name = '{}_{}_{}'.format(backbone, segmentation_body, segmentation_head)
        # Official init from torch repo.
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                if hasattr(m, 'weight'):
                    torch.nn.init.xavier_normal_(m.weight.data)

                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.fill_(1e-5)


    def forward(self, x):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        # torch.Size([2, 1024, 80, 64])
        segmentation_body_out = self.segmentation_body(backbone_out)
        # print('segmentation_body_out: ', segmentation_body_out.shape)
        # print('backbone_out: ', backbone_out[2].shape)
        # exit()
        y = self.segmentation_head(segmentation_body_out,backbone_out[2])
        # y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        if self.training:
            return y
        else:
            return y[0]


if __name__ == '__main__':
    device = torch.device('cpu')
    x = torch.zeros(2, 3, 320, 256).to(device)

    model_config = {
        'backbone': 'resnest101_aspp',
        'pretrained': True,  # backbone 是否使用imagesnet的预训练模型
        'out_channels': 3,
        'segmentation_body': {'type': 'FPEM_FFM', 'args': {'inner_channels': 256}},  # 分割头，FPN or FPEM_FFM
        'segmentation_head': {
            'type': 'espp',
            'args': {
                'num_classes':3}
        },
    }
    model = ESCNNModel(model_config=model_config).to(device)
    import time
    print(model)

    tic = time.time()
    y = model(x)
    print(time.time() - tic)
    print(y[0].shape)
    # print(model)
    # inputs = torch.random(3,320,256)
    # inputs.to(device)

    # torch.save(model.state_dict(), 'PAN.pth')
    # summary(model, input_size=(3, 320, 256))




