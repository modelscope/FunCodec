# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MS-STFT discriminator, provided here for reference."""

import typing as tp

import torchaudio
import torch
from torch import nn
from einops import rearrange

from funcodec.modules.normed_modules.conv import NormConv2d


FeatureMapType = tp.List[torch.Tensor]
LogitsType = torch.Tensor
DiscriminatorOutput = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]


def get_2d_padding(kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)


class DiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_fft (int): Size of FFT for each scale. Default: 1024
        hop_length (int): Length of hop between STFT windows for each scale. Default: 256
        kernel_size (tuple of int): Inner Conv2d kernel sizes. Default: ``(3, 9)``
        stride (tuple of int): Inner Conv2d strides. Default: ``(1, 2)``
        dilations (list of int): Inner Conv2d dilation on the time dimension. Default: ``[1, 2, 4]``
        win_length (int): Window size for each scale. Default: 1024
        normalized (bool): Whether to normalize by magnitude after stft. Default: True
        norm (str): Normalization method. Default: `'weight_norm'`
        activation (str): Activation function. Default: `'LeakyReLU'`
        activation_params (dict): Parameters to provide to the activation function.
        growth (int): Growth factor for the filters. Default: 1
    """
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                 n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024, max_filters: int = 1024,
                 filters_scale: int = 1, kernel_size: tp.Tuple[int, int] = (3, 9), dilations: tp.List = [1, 2, 4],
                 stride: tp.Tuple[int, int] = (1, 2), normalized: bool = True, norm: str = 'weight_norm',
                 activation: str = 'LeakyReLU', activation_params: dict = {'negative_slope': 0.2}):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window_fn=torch.hann_window,
            normalized=self.normalized, center=False, pad_mode=None, power=None)
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(spec_channels, self.filters, kernel_size=kernel_size, padding=get_2d_padding(kernel_size))
        )
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride,
                                         dilation=(dilation, 1), padding=get_2d_padding(kernel_size, (dilation, 1)),
                                         norm=norm))
            in_chs = out_chs
        out_chs = min((filters_scale ** (len(dilations) + 1)) * self.filters, max_filters)
        self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=(kernel_size[0], kernel_size[0]),
                                     padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                     norm=norm))
        self.conv_post = NormConv2d(out_chs, self.out_channels,
                                    kernel_size=(kernel_size[0], kernel_size[0]),
                                    padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                    norm=norm)

    def forward(self, x: torch.Tensor):
        fmap = []
        z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
        z = torch.cat([z.real, z.imag], dim=1)
        z = rearrange(z, 'b c w t -> b c t w')
        for _, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-Scale STFT (MS-STFT) discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_ffts (Sequence[int]): Size of FFT for each scale
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale
        win_lengths (Sequence[int]): Window size for each scale
        **kwargs: additional args for STFTDiscriminator
    """
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                 n_ffts: tp.List[int] = [1024, 2048, 512], hop_lengths: tp.List[int] = [256, 512, 128],
                 win_lengths: tp.List[int] = [1024, 2048, 512], **kwargs):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(filters, in_channels=in_channels, out_channels=out_channels,
                              n_fft=n_ffts[i], win_length=win_lengths[i], hop_length=hop_lengths[i], **kwargs)
            for i in range(len(n_ffts))
        ])
        self.num_discriminators = len(self.discriminators)
        self.downsample = nn.AvgPool2d(4, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor, return_intermediates: bool = True) -> tp.List:
        disc_outs = []
        for disc in self.discriminators:
            logit, fmap = disc(x)

            if return_intermediates:
                disc_outs.append((self.downsample(logit), fmap))
            else:
                disc_outs.append(self.downsample(logit))

        return disc_outs


def test():
    disc = MultiScaleSTFTDiscriminator(filters=32)
    y = torch.randn(1, 1, 24000)
    y_hat = torch.randn(1, 1, 24000)

    y_disc_r, fmap_r = disc(y)
    y_disc_gen, fmap_gen = disc(y_hat)
    # temp = None
    # for pp in y_disc_r:
    #     if temp is None:
    #         temp = torch.clamp(1-pp, min=0)
    #     else:
    #         temp += torch.clamp(1-pp, min=0)
    #     print(torch.clamp(1-pp, min=0).shape)
    # # print(y_disc_r)
    # print("ts:", temp.shape)
    # print(y_disc_r[0].shape)
    # print(y_disc_r[1].shape)
    # print(y_disc_r[2].shape)
    # print(len(fmap_r))
    # print(len(fmap_r[0]))
    print("fmap")
    for tt1 in fmap_r:
        print("tt1")
        for tt2 in tt1:
            print(tt2.shape)
    print("logits:")
    for tt1 in y_disc_r:
        print(tt1.shape)


    # c1 = torch.nn.BCELoss()
    # print(c1(y_disc_r[0], target1))
    c2 = torch.nn.BCEWithLogitsLoss()
    lossd = 0
    lossg = 0
    for tt1 in range(len(y_disc_r)):
        target1 = torch.ones(y_disc_r[tt1].shape, dtype=torch.float32)
        target0 = torch.zeros(y_disc_gen[tt1].shape, dtype=torch.float32)
        lossr = c2(y_disc_r[tt1], target1)
        lossf = c2(y_disc_gen[tt1], target0)
        lossg += lossf
        lossd += lossr + lossf
    lossd /= len(y_disc_r)
    lossf /= len(y_disc_gen)
    print(lossd)  # discriminator learning loss
    print(lossf)

    c1 = torch.nn.L1Loss()
    lossfeat = 0
    count = 0
    for tt1 in range(len(fmap_r)):
        for tt2 in range(len(fmap_r[tt1])):
            lossfeat += c1(fmap_r[tt1][tt2], fmap_gen[tt1][tt2]) / torch.mean(fmap_r[tt1][tt2])
            count += 1
    lossfeat /= count
    print("lossfeat", lossfeat)


    assert len(y_disc_r) == len(y_disc_gen) == len(fmap_r) == len(fmap_gen) == disc.num_discriminators

    assert all([len(fm) == 5 for fm in fmap_r + fmap_gen])
    assert all([list(f.shape)[:2] == [1, 32] for fm in fmap_r + fmap_gen for f in fm])
    assert all([len(logits.shape) == 4 for logits in y_disc_r + y_disc_gen])


if __name__ == '__main__':
    test()
