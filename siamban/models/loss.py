# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from siamban.core.config import cfg
from siamban.models.iou_loss import linear_iou


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)                    # origin
# -------------------------------------------------------------------------------------------------   xyl 20220607 TGFAT
#     labels_onehot = F.one_hot(label, num_classes=2).to(device=pred.device,
#                                                                        dtype=pred.dtype)
#     pt = torch.sum(labels_onehot * F.softmax(pred, dim=-1), dim=-1)
#     CE = F.cross_entropy(input=pred, target=label, reduction='none')
#     poly1 = CE + 1 * (1 - pt)
#     poly1 = poly1.mean()
#     return poly1
# -------------------------------------------------------------------------------------------------

def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    if cfg.BAN.BAN:
        diff = (pred_loc - label_loc).abs()
        diff = diff.sum(dim=1)
    else:
        diff = None
    loss = diff * loss_weight
    return loss.sum().div(pred_loc.size()[0])


def select_iou_loss(pred_loc, label_loc, label_cls):
    label_cls = label_cls.reshape(-1)
    pos = label_cls.data.eq(1).nonzero().squeeze().cuda()

    pred_loc = pred_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    pred_loc = torch.index_select(pred_loc, 0, pos)

    label_loc = label_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    label_loc = torch.index_select(label_loc, 0, pos)

    return linear_iou(pred_loc, label_loc)


# from pytorch实现polyloss https://blog.csdn.net/qq_42025868/article/details/124559449
class Poly1CrossEntropyLoss(nn.Module):
    def __init__(self,
                 num_classes: int = 2,  # xyl
                 epsilon: float = 1.0,
                 reduction: str = "none"):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(device=logits.device,
                                                                           dtype=logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        CE = F.nll_loss(input=logits, target=labels, reduction='none')
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1


class PolyLoss(torch.nn.Module):
    """
    Implementation of poly loss.
    Refers to `PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions (ICLR 2022)
    <https://arxiv.org/abs/2204.12511>
    """

    def __init__(self, num_classes=1000, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        # self.criterion = F.cross_entropy(reduction='none')
        self.num_classes = num_classes

    def forward(self, output, target):
        ce = F.cross_entropy(input=output, target=target)
        pt = F.one_hot(target, num_classes=self.num_classes) * self.softmax(output)

        return (ce + self.epsilon * (1.0 - pt.sum(dim=-1))).mean()
