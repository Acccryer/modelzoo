# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
from typing import Union
import paddle
from .mel_extractor import Wav2Mel
from .models import CFNaiveMelPE
from .paddle_interp import batch_interp_with_replacement_detach
import pathlib
import json


class DotDict(dict):
    """
    DotDict, used for config

    Example:
        # >>> config = DotDict({'a': 1, 'b': {'c': 2}}})
        # >>> config.a
        # 1
        # >>> config.b.c
        # 2
    """

    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class InferCFNaiveMelPE(paddle.nn.Layer):
    """Infer CFNaiveMelPE
    Args:
        args (DotDict): Config.
        state_dict (dict): Model state dict.
        device (str): Device. must be not None.
    """

    def __init__(self, args, state_dict, device):
        super(InferCFNaiveMelPE, self).__init__()
        self.device = device
        self.wav2mel = spawn_wav2mel(args, device=self.device)
        self.model = spawn_model(args)
        self.model.set_state_dict(state_dict)
        #self.model.to(device) # 这个api太容易炸了
        self.model.eval()
        self.args = args

    def forward(self,
                 wav: paddle.Tensor,
                 sr: Union[int, float],
                 decoder_mode: str = 'local_argmax',
                 threshold: float = 0.006
                 ) -> paddle.Tensor:
        """Infer
        Args:
            wav (paddle.Tensor): Input wav, (B, n_sample, 1).
            sr (int, float): Input wav sample rate.
            decoder_mode  (str): Decoder type. Default: "local_argmax", support "argmax" or "local_argmax".
            threshold (float): Threshold to mask. Default: 0.006.
        return: f0 (paddle.Tensor): f0 Hz, shape (B, (n_sample//hop_size + 1), 1).
        """
        with paddle.no_grad():
            #wav = wav.to(self.device)
            mel = self.wav2mel(wav, sr)
            f0 = self.model.infer(mel, decoder=decoder_mode, threshold=threshold)
        return f0  # (B, T, 1)

    def infer(self,
              wav: paddle.Tensor,
              sr: Union[int, float],
              decoder_mode: str = 'local_argmax',
              threshold: float = 0.006,
              f0_min: float = None,
              f0_max: float = None,
              interp_uv: bool = False,
              output_interp_target_length: int = None,
              ) -> paddle.Tensor:
        """Infer
        Args:
            wav (paddle.Tensor): Input wav, (B, n_sample, 1).
            sr (int, float): Input wav sample rate.
            decoder_mode (str): Decoder type. Default: "local_argmax", support "argmax" or "local_argmax".
            threshold (float): Threshold to mask. Default: 0.006.
            f0_min (float): Minimum f0. Default: None. Use in post-processing.
            f0_max (float): Maximum f0. Default: None. Use in post-processing.
            interp_uv (bool): Interpolate unvoiced frames. Default: False.
            output_interp_target_length (int): Output interpolation target length. Default: None.
        return: f0 (paddle.Tensor): f0 Hz, shape (B, (n_sample//hop_size + 1) or output_interp_target_length, 1).
        """
        # infer
        f0 = self.forward(wav, sr, decoder_mode, threshold)
        if f0_min is None:
            f0_min = self.args.model.f0_min
        # interp
        if interp_uv:
            f0 = batch_interp_with_replacement_detach((f0.squeeze(-1) < f0_min), f0.squeeze(-1)).unsqueeze(-1)
        if f0_max is not None:
            # f0[f0 > f0_max] = f0_max
            f0_max = paddle.to_tensor(f0_max,dtype=paddle.float32)
            f0 = paddle.where(f0 > f0_max, f0_max, f0)
        if output_interp_target_length is not None:
            f0 = paddle.nn.functional.interpolate(
                f0.transpose([0,2,1,]),
                size=[output_interp_target_length],
                mode='LINEAR',
                data_format='NCW'
            ).transpose([0,2,1,])
        return f0

    def f0_interp(self,
              f0: paddle.Tensor,
              f0_min: float = None,
              f0_max: float = None,
              interp_uv: bool = False,
              output_interp_target_length: int = None,
              ) -> paddle.Tensor:
        """Interp f0 produced from forward function."""
        # interp
        if interp_uv:
            f0 = batch_interp_with_replacement_detach((f0.squeeze(-1) < f0_min), f0.squeeze(-1)).unsqueeze(-1)
        if f0_max is not None:
            # f0[f0 > f0_max] = f0_max
            f0_max = paddle.to_tensor(f0_max,dtype=paddle.float32)
            f0 = paddle.where(f0 > f0_max, f0_max, f0)
        if output_interp_target_length is not None:
            f0 = paddle.nn.functional.interpolate(
                f0.transpose([0,2,1,]),
                size=[output_interp_target_length],
                mode='LINEAR',
                data_format='NCW'
            ).transpose([0,2,1,])
        return f0

    def get_hop_size(self) -> int:
        """Get hop size"""
        return self.args.mel.hop_size

    def get_hop_size_ms(self) -> float:
        """Get hop size in ms"""
        return self.args.mel.hop_size / self.args.mel.sr * 1000

    def get_model_sr(self) -> int:
        """Get model sample rate"""
        return self.args.mel.sr

    def get_mel_config(self) -> dict:
        """Get mel config"""
        return dict(self.args.mel)

    def get_device(self) -> str:
        """Get device"""
        return self.parameters[0].place

    def get_model_f0_range(self) -> dict:
        """Get model f0 range like {'f0_min': 32.70, 'f0_max': 1975.5}"""
        return {'f0_min': self.args.model.f0_min, 'f0_max': self.args.model.f0_max}


class InferCFNaiveMelPEONNX:
    """Infer CFNaiveMelPE ONNX
    Args:
        args (DotDict): Config.
        onnx_path (str): Path to onnx file.
        device (str): Device. must be not None.
    """

    def __init__(self, args, onnx_path, device):
        raise NotImplementedError


def spawn_bundled_infer_model(device: str = None) -> InferCFNaiveMelPE:
    """
    Spawn bundled infer model
    This model has been trained on our dataset and comes with the package.
    You can use it directly without anything else.
    Args:
        device (str): Device. Default: None.
    """
    file_path = pathlib.Path(__file__)
    model_path = file_path.parent / 'assets' / 'fcpe_c_v001.pdparams'
    model = spawn_infer_model_from_pt(str(model_path), device)
    return model


def spawn_infer_model_from_onnx(onnx_path: str, device: str = None) -> InferCFNaiveMelPEONNX:
    """
    Spawn infer model from onnx file
    Args:
        onnx_path (str): Path to onnx file.
        device (str): Device. Default: None.
    """
    device = get_device(device, 'paddlefcpe.tools.spawn_infer_cf_naive_mel_pe_from_onnx')
    config_path = get_config_json_in_same_path(onnx_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
        args = DotDict(config_dict)
    if (args.is_onnx is None) or (args.is_onnx is False):
        raise ValueError(f'  [ERROR] spawn_infer_model_from_onnx: this model is not onnx model.')

    if args.model.type == 'CFNaiveMelPEONNX':
        infer_model = InferCFNaiveMelPEONNX(args, onnx_path, device)
    else:
        raise ValueError(f'  [ERROR] args.model.type is {args.model.type}, but only support CFNaiveMelPEONNX')

    return infer_model


def spawn_infer_model_from_pt(pt_path: str, device: str = None) -> InferCFNaiveMelPE:
    """
    Spawn infer model from pt file
    Args:
        pt_path (str): Path to pt file.
        device (str): Device. Default: None.
    """
    device = get_device(device, 'paddlefcpe.tools.spawn_infer_cf_naive_mel_pe_from_pt')
    paddle.set_device(device)
    ckpt = paddle.load(pt_path)
    args = DotDict(ckpt['config_dict'])
    if (args.is_onnx is not None) and (args.is_onnx is True):
        raise ValueError(f'  [ERROR] spawn_infer_model_from_pt: this model is an onnx model.')

    if args.model.type == 'CFNaiveMelPE':
        infer_model = InferCFNaiveMelPE(args, ckpt['model'], device)
    else:
        raise ValueError(f'  [ERROR] args.model.type is {args.model.type}, but only support CFNaiveMelPE')

    return infer_model


def spawn_model(args: DotDict) -> CFNaiveMelPE:
    """Spawn conformer naive model"""
    if args.model.type == 'CFNaiveMelPE':
        pe_model = CFNaiveMelPE(
            input_channels=catch_none_args_must(
                args.mel.num_mels,
                func_name='paddlefcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.mel.num_mels is None',
            ),
            out_dims=catch_none_args_must(
                args.model.out_dims,
                func_name='paddlefcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.out_dims is None',
            ),
            hidden_dims=catch_none_args_must(
                args.model.hidden_dims,
                func_name='paddlefcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.hidden_dims is None',
            ),
            n_layers=catch_none_args_must(
                args.model.n_layers,
                func_name='paddlefcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.n_layers is None',
            ),
            n_heads=catch_none_args_must(
                args.model.n_heads,
                func_name='paddlefcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.n_heads is None',
            ),
            f0_max=catch_none_args_must(
                args.model.f0_max,
                func_name='paddlefcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.f0_max is None',
            ),
            f0_min=catch_none_args_must(
                args.model.f0_min,
                func_name='paddlefcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.f0_min is None',
            ),
            use_fa_norm=catch_none_args_must(
                args.model.use_fa_norm,
                func_name='paddlefcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.use_fa_norm is None',
            ),
            conv_only=catch_none_args_opti(
                args.model.conv_only,
                default=False,
                func_name='paddlefcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.conv_only is None, use default False',
            ),
            conv_dropout=catch_none_args_opti(
                args.model.conv_dropout,
                default=0.,
                func_name='paddlefcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.conv_dropout is None, use default 0.',
            ),
            atten_dropout=catch_none_args_opti(
                args.model.atten_dropout,
                default=0.,
                func_name='paddlefcpe.tools.spawn_cf_naive_mel_pe',
                warning_str='args.model.atten_dropout is None, use default 0.',
            ),
        )
    else:
        raise ValueError(f'  [ERROR] args.model.type is {args.model.type}, but only support CFNaiveMelPE')
    return pe_model


def spawn_wav2mel(args: DotDict, device: str = None) -> Wav2Mel:
    """Spawn wav2mel"""
    wav2mel = Wav2Mel(
        sr=catch_none_args_opti(
            args.mel.sr,
            default=16000,
            func_name='paddlefcpe.tools.spawn_wav2mel',
            warning_str='args.mel.sr is None',
        ),
        n_mels=catch_none_args_opti(
            args.mel.num_mels,
            default=128,
            func_name='paddlefcpe.tools.spawn_wav2mel',
            warning_str='args.mel.num_mels is None',
        ),
        n_fft=catch_none_args_opti(
            args.mel.n_fft,
            default=1024,
            func_name='paddlefcpe.tools.spawn_wav2mel',
            warning_str='args.mel.n_fft is None',
        ),
        win_size=catch_none_args_opti(
            args.mel.win_size,
            default=1024,
            func_name='paddlefcpe.tools.spawn_wav2mel',
            warning_str='args.mel.win_size is None',
        ),
        hop_length=catch_none_args_opti(
            args.mel.hop_size,
            default=160,
            func_name='paddlefcpe.tools.spawn_wav2mel',
            warning_str='args.mel.hop_size is None',
        ),
        fmin=catch_none_args_opti(
            args.mel.fmin,
            default=0,
            func_name='paddlefcpe.tools.spawn_wav2mel',
            warning_str='args.mel.fmin is None',
        ),
        fmax=catch_none_args_opti(
            args.mel.fmax,
            default=8000,
            func_name='paddlefcpe.tools.spawn_wav2mel',
            warning_str='args.mel.fmax is None',
        ),
        clip_val=catch_none_args_opti(
            args.mel.clip_val,
            default=1e-5,
            func_name='paddlefcpe.tools.spawn_wav2mel',
            warning_str='args.mel.clip_val is None',
        ),
        device=catch_none_args_opti(
            device,
            default='cpu',
            func_name='paddlefcpe.tools.spawn_wav2mel',
            warning_str='.device is None',
        ),
    )
    return wav2mel


def catch_none_args_opti(x, default, func_name, warning_str=None, level='WARN'):
    """Catch None, optional"""
    if x is None:
        if warning_str is not None:
            print(f'  [{level}] {warning_str}; use default {default}')
            print(f'  [{level}]    > call by:{func_name}')
        return default
    else:
        return x


def catch_none_args_must(x, func_name, warning_str):
    """Catch None, must"""
    level = "ERROR"
    if x is None:
        print(f'  [{level}] {warning_str}')
        print(f'  [{level}]    > call by:{func_name}')
        raise ValueError(f'  [{level}] {warning_str}')
    else:
        return x


def get_device(device: str, func_name: str) -> str:
    """Get device"""

    if device is None:
        if paddle.device.is_compiled_with_cuda():
            device = 'gpu'
        #elif paddle.backends.mps.is_available():
        #    device = 'mps'
        else:
            device = 'cpu'

        print(f'  [INFO]: Using {device} automatically.')
        print(f'  [INFO]    > call by: {func_name}')
    else:
        print(f'  [INFO]: device is not None, use {device}')
        print(f'  [INFO]    > call by:{func_name}')
        device = device

    # Check if the specified device is available, if not, switch to cpu
    if (device == 'gpu' and not paddle.device.is_compiled_with_cuda()):
        print(f'  [WARN]: Specified device ({device}) is not available, switching to cpu.')
        device = 'cpu'

    return device


def get_config_json_in_same_path(path: str) -> str:
    """Get config json in same path"""
    path = pathlib.Path(path)
    config_json = path.parent / 'config.json'
    if config_json.exists():
        return str(config_json)
    else:
        raise FileNotFoundError(f'  [ERROR] {config_json} not found.')


def bundled_infer_model_unit_test(wav_path):
    """Unit test for bundled infer model"""
    # wav_path is your wav file path
    try:
        import matplotlib.pyplot as plt
        import librosa
    except ImportError:
        print('  [UNIT_TEST] paddlefcpe.tools.spawn_infer_model_from_pt: matplotlib or librosa not found, skip test')
        exit(1)

    infer_model = spawn_bundled_infer_model(device='cpu')
    wav, sr = librosa.load(wav_path, sr=16000)
    f0 = infer_model.infer(paddle.to_tensor(wav).unsqueeze(0), sr, interp_uv=False)
    f0_interp = infer_model.infer(paddle.to_tensor(wav).unsqueeze(0), sr, interp_uv=True)
    plt.plot(f0.squeeze(-1).squeeze(0).numpy(), color='r', linestyle='-')
    plt.plot(f0_interp.squeeze(-1).squeeze(0).numpy(), color='g', linestyle='-')
    # 添加图例
    plt.legend(['f0', 'f0_interp'])
    plt.xlabel('frame')
    plt.ylabel('f0')
    plt.title('f0')
    plt.show()
