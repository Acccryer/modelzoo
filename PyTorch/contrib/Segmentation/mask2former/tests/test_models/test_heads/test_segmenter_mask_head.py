# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch_sdaa

from mmseg.models.decode_heads import SegmenterMaskTransformerHead
from .utils import _conv_has_norm, to_sdaa


def test_segmenter_mask_transformer_head():
    head = SegmenterMaskTransformerHead(
        in_channels=2,
        channels=2,
        num_classes=150,
        num_layers=2,
        num_heads=3,
        embed_dims=192,
        dropout_ratio=0.0)
    assert _conv_has_norm(head, sync_bn=False)
    head.init_weights()

    inputs = [torch.randn(1, 2, 32, 32)]
    if torch.sdaa.is_available():
        head, inputs = to_sdaa(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 32, 32)
