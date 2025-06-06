# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import torch
import torch_sdaa
from mmcv.cnn import ConvModule
from torch import Tensor, nn
from torch.nn import functional as F

from .separable_conv_module import DepthwiseSeparableConvModule


class ASPPPooling(nn.Sequential):
    """ASPP Pooling module.

    The code is adopted from
    https://github.com/pytorch/vision/blob/master/torchvision/models/
    segmentation/deeplabv3.py

    Args:
        in_channels (int): Input channels of the module.
        out_channels (int): Output channels of the module.
        conv_cfg (dict): Config dict for convolution layer. If "None",
            nn.Conv2d will be applied.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): Config dict for activation layer.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 conv_cfg: Optional[dict], norm_cfg: Optional[dict],
                 act_cfg: Optional[dict]):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                in_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for ASPP Pooling module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(
            x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """ASPP module from DeepLabV3.

    The code is adopted from
    https://github.com/pytorch/vision/blob/master/torchvision/models/
    segmentation/deeplabv3.py

    For more information about the module:
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        in_channels (int): Input channels of the module.
        out_channels (int): Output channels of the module. Default: 256.
        mid_channels (int): Output channels of the intermediate ASPP conv
            modules. Default: 256.
        dilations (Sequence[int]): Dilation rate of three ASPP conv module.
            Default: [12, 24, 36].
        conv_cfg (dict): Config dict for convolution layer. If "None",
            nn.Conv2d will be applied. Default: None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        separable_conv (bool): Whether replace normal conv with depthwise
            separable conv which is faster. Default: False.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int = 256,
                 mid_channels: int = 256,
                 dilations: Sequence[int] = (12, 24, 36),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 separable_conv: bool = False):
        super().__init__()

        if separable_conv:
            conv_module = DepthwiseSeparableConvModule
        else:
            conv_module = ConvModule

        modules = []
        modules.append(
            ConvModule(
                in_channels,
                mid_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

        for dilation in dilations:
            modules.append(
                conv_module(
                    in_channels,
                    mid_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        modules.append(
            ASPPPooling(in_channels, mid_channels, conv_cfg, norm_cfg,
                        act_cfg))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            ConvModule(
                5 * mid_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg), nn.Dropout(0.5))

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for ASPP module.

        Args:
            x (Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            Tensor: Output tensor.
        """
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
