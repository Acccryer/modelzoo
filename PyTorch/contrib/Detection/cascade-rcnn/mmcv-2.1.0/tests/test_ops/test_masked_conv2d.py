# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.utils import IS_SDAA_AVAILABLE, IS_MLU_AVAILABLE

if IS_MLU_AVAILABLE:
    torch.backends.cnnl.allow_tf32 = False
    torch.backends.mlu.matmul.allow_tf32 = False


class TestMaskedConv2d:

    @pytest.mark.parametrize('device', [
        pytest.param(
            'sdaa',
            marks=pytest.mark.skipif(
                not IS_SDAA_AVAILABLE, reason='requires SDAA support')),
        pytest.param(
            'mlu',
            marks=pytest.mark.skipif(
                not IS_MLU_AVAILABLE, reason='requires MLU support'))
    ])
    def test_masked_conv2d_all_close(self, device):
        from mmcv.ops import MaskedConv2d
        np_input = np.load(
            'tests/data/for_masked_conv2d/masked_conv2d_for_input.npy')
        np_mask = np.load(
            'tests/data/for_masked_conv2d/masked_conv2d_for_mask.npy')
        np_weight = np.load(
            'tests/data/for_masked_conv2d/masked_conv2d_for_weight.npy')
        np_bias = np.load(
            'tests/data/for_masked_conv2d/masked_conv2d_for_bias.npy')
        np_output = np.load(
            'tests/data/for_masked_conv2d/masked_conv2d_for_output.npy')
        input = torch.tensor(np_input, dtype=torch.float, device=device)
        mask = torch.tensor(np_mask, dtype=torch.float, device=device)
        weight = torch.tensor(np_weight, dtype=torch.float, device=device)
        bias = torch.tensor(np_bias, dtype=torch.float, device=device)
        conv = MaskedConv2d(3, 3, 3, 1, 1).to(device)
        conv.weight = torch.nn.Parameter(weight)
        conv.bias = torch.nn.Parameter(bias)
        output = conv(input, mask)
        assert np.allclose(output.data.cpu().numpy(), np_output, 1e-3)
