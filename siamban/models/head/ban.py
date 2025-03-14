from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamban.core.xcorr import xcorr_fast, xcorr_depthwise
# from siamban.models.backbone.wh_eca import CoordAtt


class BAN(nn.Module):
    def __init__(self):
        super(BAN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class UPChannelBAN(BAN):
    def __init__(self, feature_in=256, cls_out_channels=2):
        super(UPChannelBAN, self).__init__()

        cls_output = cls_out_channels
        loc_output = 4

        self.template_cls_conv = nn.Conv2d(feature_in, 
                feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in, 
                feature_in * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)


    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)

        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )
        # self.head = nn.Sequential(
        #         nn.Conv2d(25, hidden, kernel_size=1, bias=False),  # hidden=256 -> 25  xyl 20210611  pixel xcorr
        #         nn.BatchNorm2d(hidden),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(hidden, out_channels, kernel_size=1)
        #         )
        # self.sa = SpatialAttention()  # no used yao ## diao

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)  # torch.Size([28, 256, 25, 25])
        # feature = pixelwise_xcorr_1(search, kernel) # 像素互相关 20210520  # torch.Size([28, 25, 29, 29])  hidden=256 -> 25
        # feature = pg_xcorr(search, kernel)  # torch.Size([28, 256, 29, 29])
        # print('this is feature :', feature.shape)

        # feature = self.sa(feature) * feature  # 响应图的自空间注意力 20210520

        # img1 = tensor_to_pil(kernel)  # 画特征图 20210502 xyl
        # img1.save(r'D:\XYL\3.Object tracking\pysot-master\demo\response\kernel.jpg')
        # img2 = tensor_to_pil(search)
        # img2.save(r'D:\XYL\3.Object tracking\pysot-master\demo\response\search.jpg')
        # img3 = tensor_to_pil(feature)
        # img3.save(r'D:\XYL\3.Object tracking\pysot-master\demo\response\depthwise-response.jpg')

        out = self.head(feature)  # pixel_xcorr: in_channel = 25 not 256
        # print('this is out :', out.shape)  # torch.Size([28, 2, 25, 25]) --> torch.Size([28, 2, 29, 29])
        return out


class DepthwiseBAN(BAN):
    def __init__(self, in_channels=256, out_channels=256, cls_out_channels=2, weighted=False):
        super(DepthwiseBAN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, cls_out_channels)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        # cls_copy = cls.clone
        # cls_copy.resize(1,2,25,25)
        loc = self.loc(z_f, x_f)
        # loc_copy = loc.clone
        # loc_copy.resize(1,2,25,25)
        # print('this is cls :', cls.shape)  # torch.Size([1, 2, 29, 29])   depth: this is cls : torch.Size([1, 2, 25, 25])
        # print('this is loc :', loc.shape)  # torch.Size([1, 4, 29, 29])   depth: this is loc : torch.Size([1, 4, 25, 25])
        return cls, loc
        # return cls_copy, loc_copy

class MultiBAN(BAN):
    def __init__(self, in_channels, cls_out_channels, weighted=False):
        super(MultiBAN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('box'+str(i+2), DepthwiseBAN(in_channels[i], in_channels[i], cls_out_channels))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))
        self.loc_scale = nn.Parameter(torch.ones(len(in_channels)))
        # self.eca = eca_layer(k_size=3)                                    # xyl 20220517

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            # if idx == 4:
            #     x_f = self.eca(x_f, z_f)                                    # xyl 20220517
            box = getattr(self, 'box'+str(idx))
            c, l = box(z_f, x_f)
            cls.append(c)
            loc.append(torch.exp(l*self.loc_scale[idx-2]))

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)  # 确实是通道维度，dim=0是batch
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

#
# class eca_layer(nn.Module):
#     """Constructs a ECA module.
#
#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#
#     def __init__(self, k_size=3):
#         super(eca_layer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, z, x):
#         # feature descriptor on the global spatial information
#         # x = torch.from_numpy(x)
#         y = self.avg_pool(x)
#         # print('x.shape:', x.shape)  # x.shape: torch.Size([28, 224, 31, 31])
#         # print('y.shape:', y.shape)  # y.shape: torch.Size([28, 224, 1, 1])
#         # Two different branches of ECA module  y.squeeze(-1).transpose(-1, -2),shape ([28, 1, 224])
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         # print('y_conv.shape:', y.shape)  # y_conv.shape: torch.Size([28, 224, 1, 1])
#         # Multi-scale information fusion
#         y = self.sigmoid(y)
#         # print('y.shape', y.shape)  # y.shape torch.Size([28, 224, 1, 1])
#         y_1 = y.expand_as(z)
#         # print('y_1.shape', y_1.shape)  # y_1.shape torch.Size([28, 224, 31, 31])
#         return y * y_1