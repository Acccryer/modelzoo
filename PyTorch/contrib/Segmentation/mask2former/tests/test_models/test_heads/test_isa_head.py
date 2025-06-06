# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch_sdaa

from mmseg.models.decode_heads import ISAHead
from .utils import to_sdaa


def test_isa_head():

    inputs = [torch.randn(1, 8, 23, 23)]
    isa_head = ISAHead(
        in_channels=8,
        channels=4,
        num_classes=19,
        isa_channels=4,
        down_factor=(8, 8))
    if torch.sdaa.is_available():
        isa_head, inputs = to_sdaa(isa_head, inputs)
    output = isa_head(inputs)
    assert output.shape == (1, isa_head.num_classes, 23, 23)
