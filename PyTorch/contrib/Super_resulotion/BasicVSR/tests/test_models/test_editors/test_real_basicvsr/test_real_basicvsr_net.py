# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
import torch_sdaa

from mmagic.models.editors import RealBasicVSRNet


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-sdaa due to limited RAM.')
def test_real_basicvsr_net():
    """Test RealBasicVSR."""

    # cpu
    # is_fix_cleaning = False
    real_basicvsr = RealBasicVSRNet(is_fix_cleaning=False)

    # is_sequential_cleaning = False
    real_basicvsr = RealBasicVSRNet(
        is_fix_cleaning=True, is_sequential_cleaning=False)
    input_tensor = torch.rand(1, 5, 3, 64, 64)
    output = real_basicvsr(input_tensor)
    assert output.shape == (1, 5, 3, 256, 256)

    # is_sequential_cleaning = True, return_lq = True
    real_basicvsr = RealBasicVSRNet(
        is_fix_cleaning=True, is_sequential_cleaning=True)
    output, lq = real_basicvsr(input_tensor, return_lqs=True)
    assert output.shape == (1, 5, 3, 256, 256)
    assert lq.shape == (1, 5, 3, 64, 64)

    # gpu
    if torch.sdaa.is_available():
        # is_fix_cleaning = False
        real_basicvsr = RealBasicVSRNet(is_fix_cleaning=False).sdaa()

        # is_sequential_cleaning = False
        real_basicvsr = RealBasicVSRNet(
            is_fix_cleaning=True, is_sequential_cleaning=False).sdaa()
        input_tensor = torch.rand(1, 5, 3, 64, 64).sdaa()
        output = real_basicvsr(input_tensor)
        assert output.shape == (1, 5, 3, 256, 256)

        # is_sequential_cleaning = True, return_lq = True
        real_basicvsr = RealBasicVSRNet(
            is_fix_cleaning=True, is_sequential_cleaning=True).sdaa()
        output, lq = real_basicvsr(input_tensor, return_lqs=True)
        assert output.shape == (1, 5, 3, 256, 256)
        assert lq.shape == (1, 5, 3, 64, 64)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
