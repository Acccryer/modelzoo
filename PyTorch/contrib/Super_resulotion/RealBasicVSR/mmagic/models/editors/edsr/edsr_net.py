# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch_sdaa
import torch.nn as nn
from mmengine.model import BaseModule

from mmagic.models.archs import PixelShufflePack, ResidualBlockNoBN
from mmagic.models.utils import make_layer
from mmagic.registry import MODELS


@MODELS.register_module()
class EDSRNet(BaseModule):
    """EDSR network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_blocks (int): Block number in the trunk network. Default: 16.
        upscale_factor (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        rgb_mean (list[float]): Image mean in RGB orders.
            Default: [0.4488, 0.4371, 0.4040], calculated from DIV2K dataset.
        rgb_std (list[float]): Image std in RGB orders. In EDSR, it uses
            [1.0, 1.0, 1.0]. Default: [1.0, 1.0, 1.0].
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 num_blocks=16,
                 upscale_factor=4,
                 res_scale=1,
                 rgb_mean=[0.4488, 0.4371, 0.4040],
                 rgb_std=[1.0, 1.0, 1.0]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.num_blocks = num_blocks
        self.upscale_factor = upscale_factor

        self.mean = torch.Tensor(rgb_mean).view(1, -1, 1, 1)
        self.std = torch.Tensor(rgb_std).view(1, -1, 1, 1)

        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.body = make_layer(
            ResidualBlockNoBN,
            num_blocks,
            mid_channels=mid_channels,
            res_scale=res_scale)
        self.conv_after_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.upsample = UpsampleModule(upscale_factor, mid_channels)
        self.conv_last = nn.Conv2d(
            mid_channels, out_channels, 3, 1, 1, bias=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        self.mean = self.mean.to(x)
        self.std = self.std.to(x)

        x = (x - self.mean) / self.std
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x * self.std + self.mean

        return x


class UpsampleModule(nn.Sequential):
    """Upsample module used in EDSR.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        mid_channels (int): Channel number of intermediate features.
    """

    def __init__(self, scale, mid_channels):
        modules = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                modules.append(
                    PixelShufflePack(
                        mid_channels, mid_channels, 2, upsample_kernel=3))
        elif scale == 3:
            modules.append(
                PixelShufflePack(
                    mid_channels, mid_channels, scale, upsample_kernel=3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')

        super().__init__(*modules)
