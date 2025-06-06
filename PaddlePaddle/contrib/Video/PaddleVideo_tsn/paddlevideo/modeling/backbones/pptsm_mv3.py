# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# reference: https://arxiv.org/abs/1905.02244

from __future__ import absolute_import, division, print_function

import paddle
import paddle_sdaa
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import AdaptiveAvgPool2D, BatchNorm, Conv2D, Dropout, Linear
from paddle.regularizer import L2Decay

from ..registry import BACKBONES
from ..weight_init import weight_init_
from ...utils import load_ckpt

# Download URL of pretrained model
# MODEL_URLS = {
#     "MobileNetV3_small_x1_0":
#     "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x1_0_ssld_pretrained.pdparams",
#     "MobileNetV3_large_x1_0":
#     "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x1_0_ssld_pretrained.pdparams",
# }

MODEL_STAGES_PATTERN = {
    "MobileNetV3_small": ["blocks[0]", "blocks[2]", "blocks[7]", "blocks[10]"],
    "MobileNetV3_large":
    ["blocks[0]", "blocks[2]", "blocks[5]", "blocks[11]", "blocks[14]"]
}

# "large", "small" is just for MobinetV3_large, MobileNetV3_small respectively.
# The type of "large" or "small" config is a list. Each element(list) represents a depthwise block, which is composed of k, exp, se, act, s.
# k: kernel_size
# exp: middle channel number in depthwise block
# c: output channel number in depthwise block
# se: whether to use SE block
# act: which activation to use
# s: stride in depthwise block
NET_CONFIG = {
    "large": [
        # k, exp, c, se, act, s
        [3, 16, 16, False, "relu", 1],
        [3, 64, 24, False, "relu", 2],
        [3, 72, 24, False, "relu", 1],
        [5, 72, 40, True, "relu", 2],
        [5, 120, 40, True, "relu", 1],
        [5, 120, 40, True, "relu", 1],
        [3, 240, 80, False, "hardswish", 2],
        [3, 200, 80, False, "hardswish", 1],
        [3, 184, 80, False, "hardswish", 1],
        [3, 184, 80, False, "hardswish", 1],
        [3, 480, 112, True, "hardswish", 1],
        [3, 672, 112, True, "hardswish", 1],
        [5, 672, 160, True, "hardswish", 2],
        [5, 960, 160, True, "hardswish", 1],
        [5, 960, 160, True, "hardswish", 1],
    ],
    "small": [
        # k, exp, c, se, act, s
        [3, 16, 16, True, "relu", 2],
        [3, 72, 24, False, "relu", 2],
        [3, 88, 24, False, "relu", 1],
        [5, 96, 40, True, "hardswish", 2],
        [5, 240, 40, True, "hardswish", 1],
        [5, 240, 40, True, "hardswish", 1],
        [5, 120, 48, True, "hardswish", 1],
        [5, 144, 48, True, "hardswish", 1],
        [5, 288, 96, True, "hardswish", 2],
        [5, 576, 96, True, "hardswish", 1],
        [5, 576, 96, True, "hardswish", 1],
    ]
}
# first conv output channel number in MobileNetV3
STEM_CONV_NUMBER = 16
# last second conv output channel for "small"
LAST_SECOND_CONV_SMALL = 576
# last second conv output channel for "large"
LAST_SECOND_CONV_LARGE = 960
# last conv output channel number for "large" and "small"
LAST_CONV = 1280


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _create_act(act):
    if act == "hardswish":
        return nn.Hardswish()
    elif act == "relu":
        return nn.ReLU()
    elif act is None:
        return None
    else:
        raise RuntimeError(
            "The activation function is not supported: {}".format(act))


class MobileNetV3(nn.Layer):
    """
    MobileNetV3
    Args:
        config: list. MobileNetV3 depthwise blocks config.
        scale: float=1.0. The coefficient that controls the size of network parameters.
        class_num: int=1000. The number of classes.
        inplanes: int=16. The output channel number of first convolution layer.
        class_squeeze: int=960. The output channel number of penultimate convolution layer.
        class_expand: int=1280. The output channel number of last convolution layer.
        dropout_prob: float=0.2.  Probability of setting units to zero.
    Returns:
        model: nn.Layer. Specific MobileNetV3 model depends on args.
    """
    def __init__(self,
                 config,
                 stages_pattern,
                 scale=1.0,
                 class_num=400,
                 inplanes=STEM_CONV_NUMBER,
                 class_squeeze=LAST_SECOND_CONV_LARGE,
                 class_expand=LAST_CONV,
                 dropout_prob=0.2,
                 num_seg=8,
                 pretrained=None,
                 return_patterns=None,
                 return_stages=None):
        super().__init__()

        self.cfg = config
        self.scale = scale
        self.inplanes = inplanes
        self.class_squeeze = class_squeeze
        self.class_expand = class_expand
        self.class_num = class_num
        self.num_seg = num_seg
        self.pretrained = pretrained

        self.conv = ConvBNLayer(in_c=3,
                                out_c=_make_divisible(self.inplanes *
                                                      self.scale),
                                filter_size=3,
                                stride=2,
                                padding=1,
                                num_groups=1,
                                if_act=True,
                                act="hardswish")

        self.blocks = nn.Sequential(*[
            ResidualUnit(in_c=_make_divisible(self.inplanes * self.scale if i ==
                                              0 else self.cfg[i - 1][2] *
                                              self.scale),
                         mid_c=_make_divisible(self.scale * exp),
                         out_c=_make_divisible(self.scale * c),
                         filter_size=k,
                         stride=s,
                         use_se=se,
                         num_seg=self.num_seg,
                         act=act)
            for i, (k, exp, c, se, act, s) in enumerate(self.cfg)
        ])

        self.last_second_conv = ConvBNLayer(
            in_c=_make_divisible(self.cfg[-1][2] * self.scale),
            out_c=_make_divisible(self.scale * self.class_squeeze),
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            act="hardswish")

        self.avg_pool = AdaptiveAvgPool2D(1)

        self.last_conv = Conv2D(in_channels=_make_divisible(self.scale *
                                                            self.class_squeeze),
                                out_channels=self.class_expand,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias_attr=False)

        self.hardswish = nn.Hardswish()
        if dropout_prob is not None:
            self.dropout = Dropout(p=dropout_prob, mode="downscale_in_infer")
        else:
            self.dropout = None

        self.fc = Linear(self.class_expand, class_num)

    def init_weights(self):
        """Initiate the parameters.
        """
        if isinstance(self.pretrained, str) and self.pretrained.strip() != "":
            load_ckpt(self, self.pretrained)
        elif self.pretrained is None or self.pretrained.strip() == "":
            for layer in self.sublayers():
                if isinstance(layer, nn.Conv2D):
                    #XXX: no bias
                    weight_init_(layer, 'KaimingNormal')
                elif isinstance(layer, nn.BatchNorm2D):
                    weight_init_(layer, 'Constant', value=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.blocks(x)
        x = self.last_second_conv(x)
        x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        if self.dropout is not None:
            x = self.dropout(x)

        # feature aggregation for video
        x = paddle.reshape(x, [-1, self.num_seg, x.shape[1]])
        x = paddle.mean(x, axis=1)
        x = paddle.reshape(x, shape=[-1, self.class_expand])

        x = self.fc(x)

        return x


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 stride,
                 padding,
                 num_groups=1,
                 if_act=True,
                 act=None):
        super().__init__()

        self.conv = Conv2D(in_channels=in_c,
                           out_channels=out_c,
                           kernel_size=filter_size,
                           stride=stride,
                           padding=padding,
                           groups=num_groups,
                           bias_attr=False)
        self.bn = BatchNorm(num_channels=out_c,
                            act=None,
                            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.if_act = if_act
        self.act = _create_act(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.act(x)
        return x


class ResidualUnit(nn.Layer):
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stride,
                 use_se,
                 num_seg=8,
                 act=None):
        super().__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.if_se = use_se
        self.num_seg = num_seg

        self.expand_conv = ConvBNLayer(in_c=in_c,
                                       out_c=mid_c,
                                       filter_size=1,
                                       stride=1,
                                       padding=0,
                                       if_act=True,
                                       act=act)
        self.bottleneck_conv = ConvBNLayer(in_c=mid_c,
                                           out_c=mid_c,
                                           filter_size=filter_size,
                                           stride=stride,
                                           padding=int((filter_size - 1) // 2),
                                           num_groups=mid_c,
                                           if_act=True,
                                           act=act)
        if self.if_se:
            self.mid_se = SEModule(mid_c)
        self.linear_conv = ConvBNLayer(in_c=mid_c,
                                       out_c=out_c,
                                       filter_size=1,
                                       stride=1,
                                       padding=0,
                                       if_act=False,
                                       act=None)

    def forward(self, x):
        identity = x

        if self.if_shortcut:
            x = F.temporal_shift(x, self.num_seg, 1.0 / self.num_seg)

        x = self.expand_conv(x)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = paddle.add(identity, x)
        return x


# nn.Hardsigmoid can't transfer "slope" and "offset" in nn.functional.hardsigmoid
class Hardsigmoid(nn.Layer):
    def __init__(self, slope=0.2, offset=0.5):
        super().__init__()
        self.slope = slope
        self.offset = offset

    def forward(self, x):
        return nn.functional.hardsigmoid(x,
                                         slope=self.slope,
                                         offset=self.offset)


class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(in_channels=channel,
                            out_channels=channel // reduction,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = Conv2D(in_channels=channel // reduction,
                            out_channels=channel,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.hardsigmoid = Hardsigmoid(slope=0.2, offset=0.5)

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        return paddle.multiply(x=identity, y=x)


def PPTSM_MobileNetV3_small_x1_0(pretrained=None, **kwargs):
    """
    MobileNetV3_small_x1_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_small_x1_0` model depends on args.
    """
    model = MobileNetV3(
        config=NET_CONFIG["small"],
        scale=1.0,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        class_squeeze=LAST_SECOND_CONV_SMALL,
        pretrained=pretrained,
        **kwargs)
    return model


@BACKBONES.register()
def PPTSM_MobileNetV3(pretrained=None, **kwargs):
    """
    MobileNetV3_large_x1_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV3_large_x1_0` model depends on args.
    """
    model = MobileNetV3(
        config=NET_CONFIG["large"],
        scale=1.0,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_large"],
        class_squeeze=LAST_SECOND_CONV_LARGE,
        pretrained=pretrained,
        **kwargs)
    return model
