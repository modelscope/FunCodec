# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convolutional layers wrappers and utilities."""

import math
import typing as tp
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

from .norm import ConvLayerNorm


CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm',
                                 'time_layer_norm', 'layer_norm', 'time_group_norm'])

# self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
def apply_parametrization_norm(module: nn.Module, norm: str = 'none') -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


def get_norm_module(module: nn.Module, causal: bool = False, norm: str = 'none', **norm_kwargs) -> nn.Module:
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'layer_norm':
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        num_groups = 1
        if "num_groups" in norm_kwargs:
            num_groups = norm_kwargs.pop("num_groups")
        return nn.GroupNorm(num_groups, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                 padding_total: int = 0) -> int:
    """See `pad_for_conv1d`.
    """
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0):
    """Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))


def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'zero', value: float = 0.):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def pad2d(x: torch.Tensor, paddings: tp.Tuple[tp.Tuple[int, int], tp.Tuple[int, int]], mode: str = 'zero', value: float = 0.):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    freq_len, time_len = x.shape[-2:]
    padding_time, padding_freq = paddings
    assert min(padding_freq) >= 0 and min(padding_time) >= 0, (padding_time, padding_freq)
    if mode == 'reflect':
        max_time_pad, max_freq_pad = max(padding_time), max(padding_freq)
        extra_time_pad = max_time_pad - time_len + 1 if time_len <= max_time_pad else 0
        extra_freq_pad = max_freq_pad - freq_len + 1 if freq_len <= max_freq_pad else 0
        extra_pad = [0, extra_time_pad, 0, extra_freq_pad]
        x = F.pad(x, extra_pad)
        padded = F.pad(x, (*padding_time, *padding_freq), mode, value)
        freq_end = padded.shape[-2]-extra_freq_pad
        time_end = padded.shape[-1]-extra_time_pad
        return padded[..., :freq_end, :time_end]
    else:
        return F.pad(x, (*paddings[0], *paddings[1]), mode, value)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left: end]


def unpad2d(x: torch.Tensor, paddings: tp.Tuple[tp.Tuple[int, int], tp.Tuple[int, int]]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_time_left, padding_time_right = paddings[0]
    padding_freq_left, padding_freq_right = paddings[1]
    assert min(paddings[0]) >= 0 and min(paddings[1]) >= 0, paddings
    assert (padding_time_left + padding_time_right) <= x.shape[-1] and (padding_freq_left + padding_freq_right) <= x.shape[-2]

    freq_end = x.shape[-2] - padding_freq_right
    time_end = x.shape[-1] - padding_time_right
    return x[..., padding_freq_left: freq_end, padding_time_left: time_end]


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        #print("inputNormConv1d:", x.shape, torch.sum(x), torch.sum(torch.abs(x)), "norm:", self.norm_type, "conv:", self.conv)
        x = self.conv(x)
        #print("betweenNormConv1d:", x.shape, torch.sum(x), torch.sum(torch.abs(x)))
        if torch.isnan(torch.sum(x)).any():
            print("got nan", x.shape, self.conv)
            exit(0)
        x = self.norm(x)
        #print("outputNormConv1d:", x.shape, torch.sum(x), torch.sum(torch.abs(x)))
        return x


class NormConv2d(nn.Module):
    """Wrapper around Conv2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        #print("inputNormConv2d:", x.shape, torch.sum(x), torch.sum(torch.abs(x)))
        x = self.conv(x)
        x = self.norm(x)
        #print("outputNormConv2d:", x.shape, torch.sum(x), torch.sum(torch.abs(x)))
        return x


class NormConvTranspose1d(nn.Module):
    """Wrapper around ConvTranspose1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        #print("inputNormConvTranspose1d:", x.shape, torch.sum(x), torch.sum(torch.abs(x)))
        x = self.convtr(x)
        x = self.norm(x)
        #print("outputNormConvTranspose1d:", x.shape, torch.sum(x), torch.sum(torch.abs(x)))
        return x


class NormConvTranspose2d(nn.Module):
    """Wrapper around ConvTranspose2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)

    def forward(self, x):
        #print("inputNormConvTranspose2d:", x.shape, torch.sum(x), torch.sum(torch.abs(x)))
        x = self.convtr(x)
        x = self.norm(x)
        #print("outputNormConvTranspose2d:", x.shape, torch.sum(x), torch.sum(torch.abs(x)))
        return x


class SConv1d(nn.Module):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dilation: int = 1,
                 groups: int = 1, bias: bool = True, causal: bool = False,
                 norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {},
                 pad_mode: str = 'reflect'):
        super().__init__()
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn('SConv1d has been initialized with stride > 1 and dilation > 1'
                          f' (kernel_size={kernel_size} stride={stride}, dilation={dilation}).')
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride,
                               dilation=dilation, groups=groups, bias=bias, causal=causal,
                               norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        #print("inputSConv1d:", x.shape, torch.sum(x), torch.sum(torch.abs(x)))
        B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        padding_total = (kernel_size - 1) * dilation - (stride - 1)
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        if self.causal:
            # Left padding for causal
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        x = self.conv(x)
        #print("outputSConv1d:", x.shape, torch.sum(x), torch.sum(torch.abs(x)))
        return x


class SConvTranspose1d(nn.Module):
    """ConvTranspose1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, causal: bool = False,
                 norm: str = 'none', trim_right_ratio: float = 1.,
                 norm_kwargs: tp.Dict[str, tp.Any] = {}):
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels, out_channels, kernel_size, stride,
                                          causal=causal, norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1., \
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0. and self.trim_right_ratio <= 1.

    def forward(self, x):
        #print("inputSConvTranspose1d:", x.shape, torch.sum(x), torch.sum(torch.abs(x)))
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride

        y = self.convtr(x)

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        #print("outputSConvTranspose1d:", y.shape, torch.sum(y), torch.sum(torch.abs(y)))
        return y


def tuple_it(x, num=2):
    if isinstance(x, list):
        return tuple(x[:2])
    elif isinstance(x, int):
        return tuple([x for _ in range(num)])
    else:
        return x


class SConv2d(nn.Module):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization. Note: causal padding only make sense on time (the last) axis.
    Frequency (the second last) axis are always non-causally padded.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: tp.Union[int, tp.Tuple[int, int]],
                 stride: tp.Union[int, tp.Tuple[int, int]] = 1,
                 dilation: tp.Union[int, tp.Tuple[int, int]] = 1,
                 groups: int = 1, bias: bool = True, causal: bool = False,
                 norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {},
                 pad_mode: str = 'reflect'):
        super().__init__()
        # warn user on unusual setup between dilation and stride
        kernel_size, stride, dilation = tuple_it(kernel_size, 2), tuple_it(stride, 2), tuple_it(dilation, 2)

        if max(stride) > 1 and max(dilation) > 1:
            warnings.warn('SConv2d has been initialized with stride > 1 and dilation > 1'
                          f' (kernel_size={kernel_size} stride={stride}, dilation={dilation}).')
        self.conv = NormConv2d(in_channels, out_channels, kernel_size, stride,
                               dilation=dilation, groups=groups, bias=bias, causal=causal,
                               norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        assert len(x.shape) == 4, x.shape
        B, C, F, T = x.shape
        padding_total_list: tp.List[int] = []
        extra_padding_list: tp.List[int] = []
        for i, (kernel_size, stride, dilation) in enumerate(zip(
                self.conv.conv.kernel_size,
                self.conv.conv.stride,
                self.conv.conv.dilation
        )):
            padding_total = (kernel_size - 1) * dilation - (stride - 1)
            if i == 0:
                # no extra padding for frequency dim
                extra_padding = 0
            else:
                extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
            padding_total_list.append(padding_total)
            extra_padding_list.append(extra_padding)

        if self.causal:
            # always non-causal padding for frequency axis
            freq_after = padding_total_list[0] // 2
            freq_before = padding_total_list[0] - freq_after + extra_padding_list[0]
            # causal padding for time axis
            time_after = extra_padding_list[1]
            time_before = padding_total_list[1]
            x = pad2d(x, ((time_before, time_after), (freq_before, freq_after)), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            freq_after = padding_total_list[0] // 2
            freq_before = padding_total_list[0] - freq_after + extra_padding_list[0]
            time_after = padding_total_list[1] // 2
            time_before = padding_total_list[1] - time_after + extra_padding_list[1]
            x = pad2d(x, ((time_before, time_after), (freq_before, freq_after)), mode=self.pad_mode)
        x = self.conv(x)
        #print("outputSConv1d:", x.shape, torch.sum(x), torch.sum(torch.abs(x)))
        return x


class SConvTranspose2d(nn.Module):
    """ConvTranspose2d with some builtin handling of asymmetric or causal padding
    and normalization. Note: causal padding only make sense on time (the last) axis.
    Frequency (the second last) axis are always non-causally padded.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: tp.Union[int, tp.Tuple[int, int]],
                 stride: tp.Union[int, tp.Tuple[int, int]] = 1, causal: bool = False,
                 norm: str = 'none', trim_right_ratio: float = 1.,
                 norm_kwargs: tp.Dict[str, tp.Any] = {},
                 out_padding: tp.Union[int, tp.List[tp.Tuple[int, int]]] = 0,
                 groups: int = 1):
        super().__init__()
        self.convtr = NormConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                          causal=causal, norm=norm, norm_kwargs=norm_kwargs,
                                          groups=groups)
        if isinstance(out_padding, int):
            self.out_padding = [(out_padding, out_padding), (out_padding, out_padding)]
        else:
            self.out_padding = out_padding
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1., \
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0. and self.trim_right_ratio <= 1.

    def forward(self, x):
        #print("inputSConvTranspose1d:", x.shape, torch.sum(x), torch.sum(torch.abs(x)))
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_freq_total = kernel_size - stride
        kernel_size = self.convtr.convtr.kernel_size[1]
        stride = self.convtr.convtr.stride[1]
        padding_time_total = kernel_size - stride

        y = self.convtr(x)

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.
        (freq_out_pad_left, freq_out_pad_right) = self.out_padding[0]
        (time_out_pad_left, time_out_pad_right) = self.out_padding[1]
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            padding_freq_right = padding_freq_total // 2
            padding_freq_left = padding_freq_total - padding_freq_right
            padding_time_right = math.ceil(padding_time_total * self.trim_right_ratio)
            padding_time_left = padding_time_total - padding_time_right
            y = unpad2d(y, (
                (max(padding_time_left - time_out_pad_left, 0), max(padding_time_right - time_out_pad_right, 0)),
                (max(padding_freq_left - freq_out_pad_left, 0), max(padding_freq_right - freq_out_pad_right, 0))
            ))
        else:
            # Asymmetric padding required for odd strides
            padding_freq_right = padding_freq_total // 2
            padding_freq_left = padding_freq_total - padding_freq_right
            padding_time_right = padding_time_total // 2
            padding_time_left = padding_time_total - padding_time_right
            # y = unpad2d(y, ((padding_time_left, padding_time_right), (padding_freq_left, padding_freq_right)))
            y = unpad2d(y, (
                (max(padding_time_left - time_out_pad_left, 0), max(padding_time_right - time_out_pad_right, 0)),
                (max(padding_freq_left - freq_out_pad_left, 0), max(padding_freq_right - freq_out_pad_right, 0))
            ))
        #print("outputSConvTranspose1d:", y.shape, torch.sum(y), torch.sum(torch.abs(y)))
        return y
