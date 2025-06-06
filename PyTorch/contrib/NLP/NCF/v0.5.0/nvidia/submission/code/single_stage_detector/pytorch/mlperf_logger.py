# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

import torch
import torch_sdaa
import numpy as np
from mlperf_compliance import mlperf_log

def ssd_print(*args, **kwargs):
    barrier()
    if get_rank() == 0:
        kwargs['stack_offset'] = 2
        mlperf_log.ssd_print(*args, **kwargs)


def barrier():
    """
    Works as a temporary distributed barrier, currently pytorch
    doesn't implement barrier for NCCL backend.
    Calls all_reduce on dummy tensor and synchronizes with GPU.
    """
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(torch.sdaa.FloatTensor(1))
        torch.sdaa.synchronize()


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank

def broadcast_seeds(seed, device):
    if torch.distributed.is_initialized():
        seeds_tensor = torch.LongTensor([seed]).to(device)
        torch.distributed.broadcast(seeds_tensor, 0)
        seed = seeds_tensor.item()
    return seed

def set_seeds(args):
#   tag is not exposed for SSD
#     ssd_print(key=mlperf_log.RUN_SET_RANDOM_SEED)
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        torch.sdaa.set_device(args.local_rank)
        device = torch.device('cuda')

    # make sure that all workers has the same master seed
    args.seed = broadcast_seeds(args.seed, device)

    local_seed = (args.seed + get_rank()) % 2**32
    print(get_rank(), "Using seed = {}".format(local_seed))
    torch.manual_seed(local_seed)
    np.random.seed(seed=local_seed)
    return local_seed
