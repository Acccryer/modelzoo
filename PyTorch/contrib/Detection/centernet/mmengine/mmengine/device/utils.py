# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Optional

import torch
import torch_sdaa

try:
    import torch_npu  # noqa: F401
    import torch_npu.npu.utils as npu_utils

    # Enable operator support for dynamic shape and
    # binary operator support on the NPU.
    npu_jit_compile = bool(os.getenv('NPUJITCompile', False))
    torch.npu.set_compile_mode(jit_compile=npu_jit_compile)
    IS_NPU_AVAILABLE = hasattr(torch, 'npu') and torch.npu.is_available()
except Exception:
    IS_NPU_AVAILABLE = False

try:
    import torch_mlu  # noqa: F401
    IS_MLU_AVAILABLE = hasattr(torch, 'mlu') and torch.mlu.is_available()
except Exception:
    IS_MLU_AVAILABLE = False

try:
    import torch_dipu  # noqa: F401
    IS_DIPU_AVAILABLE = True
except Exception:
    IS_DIPU_AVAILABLE = False

try:
    import torch_musa  # noqa: F401
    IS_MUSA_AVAILABLE = True
except Exception:
    IS_MUSA_AVAILABLE = False


def get_max_sdaa_memory(device: Optional[torch.device] = None) -> int:
    """Returns the maximum GPU memory occupied by tensors in megabytes (MB) for
    a given device. By default, this returns the peak allocated memory since
    the beginning of this program.

    Args:
        device (torch.device, optional): selected device. Returns
            statistic for the current device, given by
            :func:`~torch.sdaa.current_device`, if ``device`` is None.
            Defaults to None.

    Returns:
        int: The maximum GPU memory occupied by tensors in megabytes
        for a given device.
    """
    mem = torch.sdaa.max_memory_allocated(device=device)
    mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                          dtype=torch.int,
                          device=device)
    torch.sdaa.reset_peak_memory_stats()
    return int(mem_mb.item())


def is_sdaa_available() -> bool:
    """Returns True if sdaa devices exist."""
    return torch.sdaa.is_available()


def is_npu_available() -> bool:
    """Returns True if Ascend PyTorch and npu devices exist."""
    return IS_NPU_AVAILABLE


def is_mlu_available() -> bool:
    """Returns True if Cambricon PyTorch and mlu devices exist."""
    return IS_MLU_AVAILABLE


def is_mps_available() -> bool:
    """Return True if mps devices exist.

    It's specialized for mac m1 chips and require torch version 1.12 or higher.
    """
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def is_dipu_available() -> bool:
    return IS_DIPU_AVAILABLE


def get_max_musa_memory(device: Optional[torch.device] = None) -> int:
    """Returns the maximum GPU memory occupied by tensors in megabytes (MB) for
    a given device. By default, this returns the peak allocated memory since
    the beginning of this program.

    Args:
        device (torch.device, optional): selected device. Returns
            statistic for the current device, given by
            :func:`~torch.musa.current_device`, if ``device`` is None.
            Defaults to None.

    Returns:
        int: The maximum GPU memory occupied by tensors in megabytes
        for a given device.
    """
    mem = torch.musa.max_memory_allocated(device=device)
    mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                          dtype=torch.int,
                          device=device)
    # TODO:haowen.han@mthreads.com: This function is not supported by musa yet.
    # torch.musa.reset_peak_memory_stats()
    return int(mem_mb.item())


def is_musa_available() -> bool:
    return IS_MUSA_AVAILABLE


def is_npu_support_full_precision() -> bool:
    """Returns True if npu devices support full precision training."""
    version_of_support_full_precision = 220
    return IS_NPU_AVAILABLE and npu_utils.get_soc_version(
    ) >= version_of_support_full_precision


DEVICE = 'cpu'
if is_npu_available():
    DEVICE = 'npu'
elif is_sdaa_available():
    DEVICE = 'sdaa'
elif is_mlu_available():
    DEVICE = 'mlu'
elif is_mps_available():
    DEVICE = 'mps'
elif is_dipu_available():
    DEVICE = 'dipu'
elif is_musa_available():
    DEVICE = 'musa'


def get_device() -> str:
    """Returns the currently existing device type.

    Returns:
        str: sdaa | npu | mlu | mps | musa | cpu.
    """
    return DEVICE
