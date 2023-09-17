# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamban.models.backbone.alexnet import alexnetlegacy, alexnet
from siamban.models.backbone.mobile_v2 import mobilenetv2
# from siamban.models.backbone.mobile_v2_eca import mobilenetv2
from siamban.models.backbone.repvgg import create_RepVGG_A0, create_RepVGG_B1g2
from siamban.models.backbone.resnet_atrous import resnet18, resnet34, resnet50
# from siamban.models.backbone.resnet_atrous_iAFF_triplet import resnet18, resnet34, resnet50
# from siamban.models.backbone.resnext_atrous_iAFF_triplet import resnet18, resnet34, resnet50
# from siamban.models.backbone.resnet_atrous_fca import resnet18, resnet34, resnet50
# from siamban.models.backbone.resnet_atrous_SimAM import resnet18, resnet34, resnet50

BACKBONES = {
              'alexnetlegacy': alexnetlegacy,
              'mobilenetv2': mobilenetv2,
              'resnet18': resnet18,
              'resnet34': resnet34,
              'resnet50': resnet50,
              'alexnet': alexnet,
              'repvgg-a0': create_RepVGG_A0,
              'repvgg-b1g2': create_RepVGG_B1g2
            }


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
