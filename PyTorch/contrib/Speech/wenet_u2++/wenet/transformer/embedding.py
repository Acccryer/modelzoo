# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
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
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Positonal Encoding Module."""

import math
from typing import Tuple, Union

import torch
import torch_sdaa
import torch.nn.functional as F
import numpy as np

from wenet.utils.rope_utils import precompute_freqs_cis


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 reverse: bool = False):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len,
                                dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int, torch.tensor): position offset

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """

        pos_emb = self.position_encoding(offset, x.size(1), False)
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int,
                          apply_dropout: bool = True) -> torch.Tensor:
        """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        # How to subscript a Union type:
        #   https://github.com/pytorch/pytorch/issues/69434
        if isinstance(offset, int):
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset:offset + size]
        elif isinstance(offset, torch.Tensor) and offset.dim() == 0:  # scalar
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset:offset + size]
        else:  # for batched streaming decoding on GPU
            assert torch.max(offset) + size <= self.max_len
            index = offset.unsqueeze(1) + \
                torch.arange(0, size).to(offset.device)  # B X T
            flag = index > 0
            # remove negative offset
            index = index * flag
            pos_emb = F.embedding(index, self.pe[0])  # B X T X d_model

        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        x = x * self.xscale
        pos_emb = self.position_encoding(offset, x.size(1), False)
        return self.dropout(x), self.dropout(pos_emb)


class WhisperPositionalEncoding(PositionalEncoding):
    """ Sinusoids position encoding used in openai-whisper.encoder
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 1500):
        super().__init__(d_model, dropout_rate, max_len)
        self.xscale = 1.0
        log_timescale_increment = np.log(10000) / (d_model // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment *
                                   torch.arange(d_model // 2))
        scaled_time = torch.arange(max_len)[:, np.newaxis] * \
            inv_timescales[np.newaxis, :]
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        delattr(self, "pe")
        self.register_buffer("pe", pe.unsqueeze(0))


class LearnablePositionalEncoding(PositionalEncoding):
    """ Learnable position encoding used in openai-whisper.decoder
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 448):
        super().__init__(d_model, dropout_rate, max_len)
        # NOTE(xcsong): overwrite self.pe & self.xscale
        self.pe = torch.nn.Parameter(torch.empty(1, max_len, d_model))
        self.xscale = 1.0


class NoPositionalEncoding(torch.nn.Module):
    """ No position encoding
    """

    def __init__(self, d_model: int, dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """ Just return zero vector for interface compatibility
        """
        pos_emb = torch.zeros(1, x.size(1), self.d_model).to(x.device)
        return self.dropout(x), pos_emb

    def position_encoding(self, offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        return torch.zeros(1, size, self.d_model)


class RopePositionalEncoding(PositionalEncoding):

    def __init__(self,
                 d_model: int,
                 head_dim: int,
                 dropout_rate: float,
                 max_len: int = 1500,
                 rope_theta=10000.0,
                 scale: bool = True):
        super().__init__(d_model, dropout_rate=dropout_rate, max_len=max_len)
        delattr(self, 'pe')
        self.max_len = max_len * 2
        pe = precompute_freqs_cis(head_dim, self.max_len, rope_theta)
        self.register_buffer("pe", torch.view_as_real(pe.unsqueeze(0)))
        self.dropout_rate = dropout_rate
        self.scale = scale

    def forward(
        self,
        x: torch.Tensor,
        offset: Union[int,
                      torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor]:

        pos_emb = self.position_encoding(offset, x.size(1), True)
        pos_emb = pos_emb.unsqueeze(2)  # [1,seq, 1, head_dim//2]
        # NOTE(Mddct): some model don't scale
        if self.scale:
            x = x * self.xscale
        return self.dropout(x), pos_emb

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int,
                          apply_dropout: bool = True) -> torch.Tensor:

        pe = torch.view_as_complex(self.pe)
        if isinstance(offset, int):
            assert offset + size <= self.max_len
            pos_emb = pe[:, offset:offset + size]
        else:
            assert torch.max(offset) + size <= self.max_len
            index = offset.unsqueeze(1) + torch.arange(0, size).to(
                offset.device)  # B X T
            flag = index > 0
            # remove negative offset
            index = index * flag
            pos_emb = F.embedding(index, pe[0])  # B X T X head_dim//2
        if apply_dropout:
            # NOTE(Mddct) dropout don't suuport complex float for pos_emb
            pos_emb = self.dropout_complex(pos_emb)
        return pos_emb

    def dropout_complex(self, x):
        mask = torch.nn.functional.dropout(
            torch.ones_like(x.real),
            training=self.training,
            p=self.dropout_rate,
        )
        return x * mask
