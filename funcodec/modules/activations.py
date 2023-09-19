import torch
from torch import nn


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


def get_activation(activation: str = None, channels=None, **kwargs):
    if activation.lower() == "snake":
        assert channels is not None, "Snake activation needs channel number."
        return Snake1d(channels=channels)
    else:
        act = getattr(nn, activation)
        return act(**kwargs)
