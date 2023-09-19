from torch import nn
import torch.nn.functional as F
from itertools import cycle
import functools
import torch
from einops import rearrange, reduce, pack, unpack
from typing import List, Tuple
from local_attention import LocalMHA
from local_attention.transformer import FeedForward, DynamicPositionBias


def round_down_nearest_multiple(num, divisor):
    return num // divisor * divisor


def curtail_to_multiple(t, mult):
    data_len = t.shape[-1]
    return t[..., :round_down_nearest_multiple(data_len, mult)]


# helper functions
def exists(val):
    return val is not None


# better sequential
def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))


# autoregressive squeeze excitation
# https://arxiv.org/abs/1709.01507
class SqueezeExcite(nn.Module):
    def __init__(self, dim, reduction_factor=4, dim_minimum=8):
        super().__init__()
        dim_inner = max(dim_minimum, dim // reduction_factor)
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim_inner, 1),
            nn.SiLU(),
            nn.Conv1d(dim_inner, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        seq, device = x.shape[-2], x.device

        # cumulative mean - since it is autoregressive
        cum_sum = x.cumsum(dim=-2)
        denom = torch.arange(1, seq + 1, device=device).float()
        cum_mean = cum_sum / rearrange(denom, 'n -> n 1')

        # glu gate
        gate = self.net(cum_mean)

        return x * gate


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class CausalConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, **kwargs):
        super().__init__()
        kernel_size = kernel_size
        dilation = kwargs.get('dilation', 1)
        self.causal_padding = dilation * (kernel_size - 1)

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.causal_padding, 0))
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride, **kwargs):
        super().__init__()
        self.upsample_factor = stride
        self.padding = kernel_size - 1
        self.conv = nn.ConvTranspose1d(chan_in, chan_out, kernel_size, stride, **kwargs)

    def forward(self, x):
        n = x.shape[-1]

        out = self.conv(x)
        out = out[..., :(n * self.upsample_factor)]

        return out


def ResidualUnit(chan_in, chan_out, dilation, kernel_size=7, squeeze_excite=False):
    return Residual(Sequential(
        CausalConv1d(chan_in, chan_out, kernel_size, dilation=dilation),
        nn.ELU(),
        CausalConv1d(chan_out, chan_out, 1),
        nn.ELU(),
        SqueezeExcite(chan_out) if squeeze_excite else None
    ))


def EncoderBlock(chan_in, chan_out, stride, cycle_dilations=(1, 3, 9), squeeze_excite=False):
    it = cycle(cycle_dilations)
    residual_unit = functools.partial(ResidualUnit, squeeze_excite=squeeze_excite)

    return nn.Sequential(
        residual_unit(chan_in, chan_in, next(it)),
        residual_unit(chan_in, chan_in, next(it)),
        residual_unit(chan_in, chan_in, next(it)),
        CausalConv1d(chan_in, chan_out, 2 * stride, stride=stride)
    )


class LocalTransformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            heads,
            window_size,
            dynamic_pos_bias=False,
            **kwargs
    ):
        super().__init__()
        self.window_size = window_size
        self.layers = nn.ModuleList([])

        self.pos_bias = None
        if dynamic_pos_bias:
            self.pos_bias = DynamicPositionBias(dim=dim // 2, heads=heads)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LocalMHA(dim=dim, heads=heads, qk_rmsnorm=True, window_size=window_size,
                         use_rotary_pos_emb=not dynamic_pos_bias, use_xpos=True, **kwargs),
                FeedForward(dim=dim)
            ]))

    def forward(self, x):
        w = self.window_size

        attn_bias = self.pos_bias(w, w * 2) if exists(self.pos_bias) else None

        for attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x

        return x


class SoundStreamConvEncoder(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            channels: int = 32,
            strides: Tuple = (2, 4, 5, 8),
            channel_mults: Tuple = (2, 4, 8, 16),
            dilations: Tuple = (1, 3, 9),
            output_channels: int = 512,
            use_local_attn: bool = True,
            attn_window_size: int = 128,
            attn_dim_head: int = 64,
            attn_heads: int = 8,
            attn_depth: int = 1,
            attn_xpos_scale_base: int = None,
            attn_dynamic_pos_bias: bool = False,
            squeeze_excite: bool = False,
    ):
        super().__init__()
        self.input_channel = input_size
        self.output_channels = output_channels

        self.strides = strides
        layer_channels = tuple(map(lambda t: t * channels, channel_mults))
        layer_channels = (channels, *layer_channels)
        chan_in_out_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        encoder_blocks = []

        for ((chan_in, chan_out), layer_stride) in zip(chan_in_out_pairs, strides):
            encoder_blocks.append(
                EncoderBlock(
                    chan_in,
                    chan_out,
                    layer_stride,
                    dilations,
                    squeeze_excite
                )
            )

        self.encoder = nn.Sequential(
            CausalConv1d(input_size, channels, 7),
            *encoder_blocks,
            CausalConv1d(layer_channels[-1], output_channels, 3)
        )

        attn_kwargs = dict(
            dim=output_channels,
            dim_head=attn_dim_head,
            heads=attn_heads,
            depth=attn_depth,
            window_size=attn_window_size,
            xpos_scale_base=attn_xpos_scale_base,
            dynamic_pos_bias=attn_dynamic_pos_bias,
            prenorm=True,
            causal=True
        )

        # local (windowed) and causal transformer with `depth` layers.
        self.encoder_attn = LocalTransformer(**attn_kwargs) if use_local_attn else None

    @property
    def seq_len_multiple_of(self):
        return functools.reduce(lambda x, y: x * y, self.strides)

    def forward(
            self,
            x,
    ):
        # clip the length of inputs into multiple of strides
        x = curtail_to_multiple(x, self.seq_len_multiple_of)

        x = self.encoder(x)
        # permute for transformer layers
        x = rearrange(x, 'b c n -> b n c')

        if exists(self.encoder_attn):
            x = self.encoder_attn(x)

        return x

    def output_size(self):
        return self.output_channels
