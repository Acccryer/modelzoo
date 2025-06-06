# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/huggingface/diffusers

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch_sdaa
import torch.nn.functional as F
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from torch import nn

from mmagic.models.editors.ddpm.attention import GEGLU, ApproximateGELU
from .attention_3d import CrossAttention


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
    """Output of TemporalTransformer3DModel."""
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def get_motion_module(in_channels, motion_module_type: str,
                      motion_module_kwargs: dict):
    """Get motion module."""
    if motion_module_type == 'Vanilla':
        return VanillaTemporalModule(
            in_channels=in_channels,
            **motion_module_kwargs,
        )
    else:
        raise ValueError


class VanillaTemporalModule(nn.Module):
    """Module which uses transformer to handle 3d motion."""

    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        num_transformer_block=2,
        attention_block_types=('Temporal_Self', 'Temporal_Self'),
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        temporal_attention_dim_div=1,
        zero_initialize=True,
    ):
        super().__init__()
        temp_pos_max_len = temporal_position_encoding_max_len
        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads //
            temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temp_pos_max_len,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(
                self.temporal_transformer.proj_out)

    def forward(self,
                input_tensor,
                temb,
                encoder_hidden_states,
                attention_mask=None,
                anchor_frame_idx=None):
        """forward with sample."""
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(hidden_states,
                                                  encoder_hidden_states,
                                                  attention_mask)

        output = hidden_states
        return output


class TemporalTransformer3DModel(nn.Module):
    """Module which uses implement 3D Transformer."""

    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        attention_block_types=(
            'Temporal_Self',
            'Temporal_Self',
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn='geglu',
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            eps=1e-6,
            affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        temp_pos_max_len = temporal_position_encoding_max_len
        self.transformer_blocks = nn.ModuleList([
            TemporalTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                attention_block_types=attention_block_types,
                dropout=dropout,
                norm_num_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
                upcast_attention=upcast_attention,
                cross_frame_attention_mode=cross_frame_attention_mode,
                temporal_position_encoding=temporal_position_encoding,
                temporal_position_encoding_max_len=temp_pos_max_len,
            ) for d in range(num_layers)
        ])
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(self,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None):
        """forward with hidden states, encoder_hidden_states and
        attention_mask."""

        assert hidden_states.dim(
        ) == 5, f'{"Expected hidden_states to have ndim=5, "}'
        f'but got ndim={hidden_states.dim()}.'
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, 'b c f h w -> (b f) c h w')

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * weight, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length)

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height,
                                              weight, inner_dim).permute(
                                                  0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, '(b f) c h w -> b c f h w', f=video_length)

        return output


class TemporalTransformerBlock(nn.Module):
    """Module which is a component of Temporal 3D Transformer."""

    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types=(
            'Temporal_Self',
            'Temporal_Self',
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn='geglu',
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        attention_blocks = []
        norms = []
        temp_pos_max_len = temporal_position_encoding_max_len
        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split('_')[0],
                    cross_attention_dim=cross_attention_dim
                    if block_name.endswith('_Cross') else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temp_pos_max_len,
                ))
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(
            dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                video_length=None):
        """forward with hidden states, encoder_hidden_states and
        attention_mask."""
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states
                if attention_block.is_cross_attention else None,
                video_length=video_length,
            ) + hidden_states

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output


class PositionalEncoding(nn.Module):
    """a implementation of PositionEncoding."""

    def __init__(self, d_model, dropout=0., max_len=24):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """forward function."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class VersatileAttention(CrossAttention):
    """a implementation of VersatileAttention."""

    def __init__(self,
                 attention_mode=None,
                 cross_frame_attention_mode=None,
                 temporal_position_encoding=False,
                 temporal_position_encoding_max_len=24,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert attention_mode == 'Temporal'

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs['cross_attention_dim'] is not None

        self.pos_encoder = PositionalEncoding(
            kwargs['query_dim'],
            dropout=0.,
            max_len=temporal_position_encoding_max_len) if (
                temporal_position_encoding
                and attention_mode == 'Temporal') else None

        self._use_memory_efficient_attention_xformers = False

    def extra_repr(self):
        """return module information."""
        return f'(Module Info) Attention_Mode: {self.attention_mode},\
            Is_Cross_Attention: {self.is_cross_attention}'

    def reshape_heads_to_batch_dim(self, tensor):
        """reshape heads num to batch dim."""
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size,
                                dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size,
                                                    seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        """reshape batch dim to heads num."""
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len,
                                dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size,
                                                    seq_len, dim * head_size)
        return tensor

    def _memory_efficient_attention_xformers(self, query, key, value,
                                             attention_mask):
        """use xformers to save memory."""
        # TODO attention_mask
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def forward(self,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                video_length=None):
        """forward with hidden states, encoder_hidden_states and
        attention_mask."""
        batch_size, sequence_length, _ = hidden_states.shape

        if self.attention_mode == 'Temporal':
            d = hidden_states.shape[1]
            hidden_states = rearrange(
                hidden_states, '(b f) d c -> (b d) f c', f=video_length)

            if self.pos_encoder is not None:
                hidden_states = self.pos_encoder(hidden_states)

            encoder_hidden_states = repeat(
                encoder_hidden_states, 'b n c -> (b d) n c', d=d
            ) if encoder_hidden_states is not None else encoder_hidden_states
        else:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(
                1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states \
            if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(
                    attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(
                    self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(
                query, key, value, attention_mask)
            # Some versions of xformers return output in fp32,
            # cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[
                    0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value,
                                                attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value,
                                                       sequence_length, dim,
                                                       attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        if self.attention_mode == 'Temporal':
            hidden_states = rearrange(
                hidden_states, '(b d) f c -> (b f) d c', d=d)

        return hidden_states


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*):
        The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4):
        The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
        Activation function to be used in feed-forward.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = 'geglu',
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == 'gelu':
            act_fn = GELU(dim, inner_dim)
        elif activation_fn == 'geglu':
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == 'geglu-approximate':
            act_fn = ApproximateGELU(dim, inner_dim)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class GELU(nn.Module):
    r"""
    GELU activation function
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def gelu(self, gate):
        if gate.device.type != 'mps':
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states
