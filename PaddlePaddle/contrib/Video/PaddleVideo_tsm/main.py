# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import argparse
import random

import numpy as np
import paddle
import paddle_sdaa

from paddlevideo.tasks import (test_model, train_dali, train_model,
                               train_model_multigrid)
from paddlevideo.utils import get_config, get_dist_info


def parse_args():
    parser = argparse.ArgumentParser("PaddleVideo train script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    parser.add_argument('-o',
                        '--override',
                        action='append',
                        default=[],
                        help='config options to be overridden')
    parser.add_argument('--test',
                        action='store_true',
                        help='whether to test a model')
    parser.add_argument('--train_dali',
                        action='store_true',
                        help='whether to use dali to speed up training')
    parser.add_argument('--multigrid',
                        action='store_true',
                        help='whether to use multigrid training')
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        help='weights for finetuning or testing')
    parser.add_argument('--fleet',
                        action='store_true',
                        help='whether to use fleet run distributed training')
    parser.add_argument('--amp',
                        action='store_true',
                        help='whether to open amp training.')
    parser.add_argument(
        '--amp_level',
        type=str,
        default=None,
        help="optimize level when open amp training, can only be 'O1' or 'O2'.")
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--seed',
        type=int,
        default=1234,
        help='fixed all random seeds when the program is running')
    parser.add_argument(
        '--max_iters',
        type=int,
        default=None,
        help='max iterations when training(this arg only used in test_tipc)')
    parser.add_argument(
        '-p',
        '--profiler_options',
        type=str,
        default=None,
        help='The option of profiler, which should be in format '
        '\"key1=value1;key2=value2;key3=value3\".')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = get_config(args.config, overrides=args.override)

    # enable to use npu if paddle is built with npu
    if paddle.is_compiled_with_custom_device('npu') :
        cfg.__setattr__("use_npu", True)
    elif paddle.device.is_compiled_with_xpu():
        cfg.__setattr__("use_xpu", True)

    # set seed if specified
    seed = args.seed
    if seed is not None:
        assert isinstance(
            seed, int), f"seed must be a integer when specified, but got {seed}"
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    # set amp_level if amp is enabled
    if args.amp:
        if args.amp_level is None:
            args.amp_level = 'O1'  # set defaualt amp_level to 'O1'
        else:
            assert args.amp_level in [
                'O1', 'O2'
            ], f"amp_level must be 'O1' or 'O2' when amp enabled, but got {args.amp_level}."

    _, world_size = get_dist_info()
    parallel = world_size != 1
    if parallel:
        paddle.distributed.init_parallel_env()

    if args.test:
        test_model(cfg, weights=args.weights, parallel=parallel)
    elif args.train_dali:
        train_dali(cfg, weights=args.weights, parallel=parallel)
    elif args.multigrid:
        train_model_multigrid(cfg,
                              world_size=world_size,
                              validate=args.validate)
    else:
        train_model(cfg,
                    weights=args.weights,
                    parallel=parallel,
                    validate=args.validate,
                    use_fleet=args.fleet,
                    use_amp=args.amp,
                    amp_level=args.amp_level,
                    max_iters=args.max_iters,
                    profiler_options=args.profiler_options)


if __name__ == '__main__':
    main()
