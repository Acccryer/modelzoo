# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch_sdaa

from mmagic.models.archs import SimpleGatedConvModule
from mmagic.models.editors import ContextualAttentionNeck


def test_deepfill_contextual_attention_neck():
    # TODO: add unittest for contextual attention module
    neck = ContextualAttentionNeck(in_channels=128)
    x = torch.rand((2, 128, 64, 64))
    mask = torch.zeros((2, 1, 64, 64))
    mask[..., 20:100, 23:90] = 1.

    res, offset = neck(x, mask)

    assert res.shape == (2, 128, 64, 64)
    assert offset.shape == (2, 32, 32, 32, 32)

    if torch.sdaa.is_available():
        neck.sdaa()
        res, offset = neck(x.sdaa(), mask.sdaa())

        assert res.shape == (2, 128, 64, 64)
        assert offset.shape == (2, 32, 32, 32, 32)

        neck = ContextualAttentionNeck(
            in_channels=128, conv_type='gated_conv').sdaa()
        res, offset = neck(x.sdaa(), mask.sdaa())
        assert res.shape == (2, 128, 64, 64)
        assert offset.shape == (2, 32, 32, 32, 32)
        assert isinstance(neck.conv1, SimpleGatedConvModule)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
