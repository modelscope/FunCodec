import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from typing import Dict, Any


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


class ConvDiscriminator(nn.Module):
    def __init__(
            self,
            in_channels=1,
            channels=16,
            layers=4,
            groups=4,
            chan_max=1024,
    ):
        super().__init__()
        self.init_conv = nn.Conv1d(in_channels, channels, 7)
        self.conv_layers = nn.ModuleList([])

        curr_channels = channels

        for _ in range(layers):
            chan_out = min(curr_channels * 4, chan_max)

            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(curr_channels, chan_out, 8, stride=4, padding=4, groups=groups),
                leaky_relu()
            ))

            curr_channels = chan_out

        self.final_conv = nn.Sequential(
            nn.Conv1d(curr_channels, curr_channels, 3),
            leaky_relu(),
            nn.Conv1d(curr_channels, 1, 1),
        )

    def forward(self, x, return_intermediates=True):
        x = self.init_conv(x)

        intermediates = []

        for layer in self.conv_layers:
            x = layer(x)
            intermediates.append(x)

        out = self.final_conv(x)

        if not return_intermediates:
            return out

        return out, intermediates


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(
            self,
            in_channels=1,
            disc_multi_scales=(1, 0.5, 0.25),
            discriminator_params: Dict[str, Any] = dict(
                channels=16,
                layers=4,
                groups=4,
                chan_max=1024
            )
    ):
        super().__init__()
        self.disc_multi_scales = disc_multi_scales
        self.discriminators = nn.ModuleList([
            ConvDiscriminator(in_channels=in_channels, **discriminator_params)
            for _ in range(len(disc_multi_scales))
        ])
        disc_rel_factors = [int(s1 / s2) for s1, s2 in zip(disc_multi_scales[:-1], disc_multi_scales[1:])]
        self.downsamples = nn.ModuleList(
            [nn.Identity()] +
            [nn.AvgPool1d(2 * factor, stride=factor, padding=factor)
             for factor in disc_rel_factors]
        )

    def forward(self, x, return_intermediates=True):
        disc_outs = []
        for discr, downsample in zip(self.discriminators, self.downsamples):
            scaled_x = downsample(x)

            logits, intermediates = discr(scaled_x)

            if return_intermediates:
                disc_outs.append((logits, intermediates))
            else:
                disc_outs.append(logits)

        return disc_outs


class ModReLU(nn.Module):
    """
    https://arxiv.org/abs/1705.09792
    https://github.com/pytorch/pytorch/issues/47052#issuecomment-718948801
    """

    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return F.relu(torch.abs(x) + self.b) * torch.exp(1.j * torch.angle(x))


class ComplexConv2d(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            kernel_size,
            stride=1,
            padding=0
    ):
        super().__init__()
        conv = nn.Conv2d(dim, dim_out, kernel_size, dtype=torch.complex64)
        self.weight = nn.Parameter(torch.view_as_real(conv.weight))
        self.bias = nn.Parameter(torch.view_as_real(conv.bias))

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        weight, bias = map(torch.view_as_complex, (self.weight, self.bias))

        x = x.to(weight.dtype)
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)


def ComplexSTFTResidualUnit(chan_in, chan_out, strides):
    kernel_sizes = tuple(map(lambda t: t + 2, strides))
    paddings = tuple(map(lambda t: t // 2, kernel_sizes))

    return nn.Sequential(
        ComplexConv2d(chan_in, chan_in, 3, padding=1),
        ModReLU(),
        ComplexConv2d(chan_in, chan_out, kernel_sizes, stride=strides, padding=paddings)
    )


class ComplexSTFTDiscriminator(nn.Module):
    def __init__(
            self,
            *,
            in_channels=1,
            channels=32,
            strides=((1, 2), (2, 2), (1, 2), (2, 2), (1, 2), (2, 2)),
            chan_mults=(1, 2, 4, 4, 8, 8),
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            stft_normalized=False,
            logits_abs=True
    ):
        super().__init__()
        self.init_conv = ComplexConv2d(in_channels, channels, 7, padding=3)

        layer_channels = tuple(map(lambda mult: mult * channels, chan_mults))
        layer_channels = (channels, *layer_channels)
        layer_channels_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        curr_channels = channels

        self.layers = nn.ModuleList([])

        for layer_stride, (chan_in, chan_out) in zip(strides, layer_channels_pairs):
            self.layers.append(ComplexSTFTResidualUnit(chan_in, chan_out, layer_stride))

        self.final_conv = ComplexConv2d(layer_channels[-1], 1, (16, 1))  # todo: remove hardcoded 16

        # stft settings

        self.stft_normalized = stft_normalized

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # how to output the logits into real space
        self.logits_abs = logits_abs

    def forward(self, x, return_intermediates=True):
        x = rearrange(x, 'b 1 n -> b n')

        '''
        reference: The content of the paper( https://arxiv.org/pdf/2107.03312.pdf)is as follows:
        The STFT-based discriminator is illustrated in Figure 4
        and operates on a single scale, computing the STFT with a
        window length of W = 1024 samples and a hop length of
        H = 256 samples
        '''

        x = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            normalized=self.stft_normalized,
            return_complex=True
        )

        x = rearrange(x, 'b ... -> b 1 ...')

        intermediates = []

        x = self.init_conv(x)

        intermediates.append(x)

        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)

        complex_logits = self.final_conv(x)

        if self.logits_abs:
            complex_logits = complex_logits.abs()
        else:
            complex_logits = torch.view_as_real(complex_logits)

        if not return_intermediates:
            return complex_logits

        return complex_logits, intermediates
