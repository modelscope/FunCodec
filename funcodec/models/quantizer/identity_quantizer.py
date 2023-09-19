import torch


# This class should be only used for debug.
class IdentityQuantizer(torch.nn.Module):
    def __init__(
            self,
            input_size: int = 512,
            **kwargs
    ):
        super().__init__()
        self.code_dim = input_size
        self.input_size = input_size
        self.register_buffer('zero', torch.tensor([0.]), persistent=False)

    def forward(
            self,
            x,
    ):
        return x, None, self.zero

    def output_size(self):
        return self.code_dim
