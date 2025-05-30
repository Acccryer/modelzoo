# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
from os.path import dirname, join
from unittest.mock import MagicMock

import pytest
import torch
import torch_sdaa
from mmengine import Config
from mmengine.optim import OptimWrapper

from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules


# TODO: this test is same as `test_two_stage_encoder_decoder.py`
@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-sdaa due to limited RAM.')
def test_two_stage_inpaintor():
    register_all_modules()

    config_file = join(dirname(__file__), '../../configs', 'two_stage_test.py')
    cfg = Config.fromfile(config_file)

    inpaintor = MODELS.build(cfg.model)

    assert inpaintor.__class__.__name__ == 'TwoStageInpaintor'

    if torch.sdaa.is_available():
        inpaintor.sdaa()

    # check architecture
    assert inpaintor.stage1_loss_type == ('loss_l1_hole', 'loss_l1_valid',
                                          'loss_composed_percep', 'loss_tv')
    assert inpaintor.stage2_loss_type == ('loss_l1_hole', 'loss_l1_valid',
                                          'loss_gan')
    assert inpaintor.with_l1_hole_loss
    assert inpaintor.with_l1_valid_loss
    assert inpaintor.with_composed_percep_loss
    assert not inpaintor.with_out_percep_loss
    assert inpaintor.with_gan

    inpaintor._modules['loss_percep'] = MagicMock(
        return_value=(torch.Tensor([1]), torch.Tensor([2])))

    # prepare data
    gt_img = torch.rand((3, 256, 256))
    mask = torch.zeros((1, 256, 256))
    mask[..., 50:180, 60:170] = 1.
    masked_img = gt_img.unsqueeze(0) * (1. - mask)
    mask_bbox = [100, 100, 110, 110]
    data_batch = {
        'inputs':
        masked_img,
        'data_samples':
        [DataSample(
            mask=mask,
            mask_bbox=mask_bbox,
            gt_img=gt_img,
        )]
    }

    optim_g = torch.optim.Adam(inpaintor.generator.parameters(), lr=0.0001)
    optim_d = torch.optim.Adam(inpaintor.disc.parameters(), lr=0.0001)
    optims = dict(generator=OptimWrapper(optim_g), disc=OptimWrapper(optim_d))

    # check train_step with standard two_stage model
    for i in range(5):
        log_vars = inpaintor.train_step(data_batch, optims)

        if i % 2 == 0:
            assert 'fake_loss' in log_vars
            assert 'real_loss' in log_vars
            assert 'loss_disc_shift' in log_vars
            assert 'loss' in log_vars
        else:
            assert 'fake_loss' in log_vars
            assert 'real_loss' in log_vars
            assert 'loss_disc_shift' in log_vars
            assert 'loss' in log_vars
            assert 'stage1_loss_l1_hole' in log_vars
            assert 'stage1_loss_l1_valid' in log_vars
            assert 'stage2_loss_l1_hole' in log_vars
            assert 'stage2_loss_l1_valid' in log_vars

    # check for forward_test
    data = inpaintor.data_preprocessor(data_batch, True)
    data_inputs, data_sample = data['inputs'], data['data_samples']
    output = inpaintor.forward_test(data_inputs, data_sample)
    prediction = output
    assert 'fake_res' in prediction
    assert 'fake_img' in prediction
    assert 'pred_img' in prediction
    assert prediction.pred_img.shape == (1, 3, 256, 256)

    # check for gp_loss
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.model.disc_input_with_mask = False
    cfg_copy.model.disc.in_channels = 3
    cfg_copy.model.loss_gp = dict(type='GradientPenaltyLoss', loss_weight=10.)
    inpaintor = MODELS.build(cfg_copy.model)
    assert inpaintor.with_gp_loss

    log_vars = inpaintor.train_step(data_batch, optims)
    assert 'real_loss' in log_vars
    assert 'stage1_loss_l1_hole' in log_vars
    assert 'stage1_loss_l1_valid' in log_vars
    assert 'stage2_loss_l1_hole' in log_vars
    assert 'stage2_loss_l1_valid' in log_vars


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
