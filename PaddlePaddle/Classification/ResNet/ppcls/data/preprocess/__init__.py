# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted to tecorigin hardware

from ppcls.data.preprocess.ops.autoaugment import ImageNetPolicy as RawImageNetPolicy
from ppcls.data.preprocess.ops.randaugment import RandAugment as RawRandAugment
from ppcls.data.preprocess.ops.timm_autoaugment import RawTimmAutoAugment
from ppcls.data.preprocess.ops.cutout import Cutout

from ppcls.data.preprocess.ops.hide_and_seek import HideAndSeek
from ppcls.data.preprocess.ops.random_erasing import RandomErasing
from ppcls.data.preprocess.ops.grid import GridMask

from ppcls.data.preprocess.ops.operators import DecodeImage
from ppcls.data.preprocess.ops.operators import ResizeImage
from ppcls.data.preprocess.ops.operators import CropImage
from ppcls.data.preprocess.ops.operators import RandCropImage
from ppcls.data.preprocess.ops.operators import RandCropImageV2
from ppcls.data.preprocess.ops.operators import RandFlipImage
from ppcls.data.preprocess.ops.operators import NormalizeImage
from ppcls.data.preprocess.ops.operators import ToCHWImage
from ppcls.data.preprocess.ops.operators import AugMix
from ppcls.data.preprocess.ops.operators import Pad
from ppcls.data.preprocess.ops.operators import ToTensor
from ppcls.data.preprocess.ops.operators import Normalize
from ppcls.data.preprocess.ops.operators import RandomHorizontalFlip
from ppcls.data.preprocess.ops.operators import CropWithPadding
from ppcls.data.preprocess.ops.operators import RandomInterpolationAugment
from ppcls.data.preprocess.ops.operators import ColorJitter
from ppcls.data.preprocess.ops.operators import RandomCropImage
from ppcls.data.preprocess.ops.operators import RandomRotation
from ppcls.data.preprocess.ops.operators import Padv2

from ppcls.data.preprocess.batch_ops.batch_operators import MixupOperator, CutmixOperator, OpSampler, FmixOperator
from ppcls.data.preprocess.batch_ops.batch_operators import MixupCutmixHybrid

import numpy as np
from PIL import Image
import random


def transform(data, ops=[]):
    """ transform """
    for op in ops:
        data = op(data)
    return data


class AutoAugment(RawImageNetPolicy):
    """ ImageNetPolicy wrapper to auto fit different img types """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = np.ascontiguousarray(img)
            img = Image.fromarray(img)

        img = super().__call__(img)

        if isinstance(img, Image.Image):
            img = np.asarray(img)

        return img


class RandAugment(RawRandAugment):
    """ RandAugment wrapper to auto fit different img types """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = np.ascontiguousarray(img)
            img = Image.fromarray(img)

        img = super().__call__(img)

        if isinstance(img, Image.Image):
            img = np.asarray(img)

        return img


class TimmAutoAugment(RawTimmAutoAugment):
    """ TimmAutoAugment wrapper to auto fit different img tyeps. """

    def __init__(self, prob=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob = prob

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = np.ascontiguousarray(img)
            img = Image.fromarray(img)
        if random.random() < self.prob:
            img = super().__call__(img)
        if isinstance(img, Image.Image):
            img = np.asarray(img)

        return img
