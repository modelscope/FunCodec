import torch
from funcodec.modules.vector_quantize_pytorch.residual_vq import ResidualVQ


class ResidualQuantizer(torch.nn.Module):
    def __init__(
            self,
            input_size: int = 512,
            codebook_size: int = 1024,
            num_quantizers: int = 8,
            commitment_weight: float = 1.,
            ema_decay: float = 0.95,
            quantize_dropout_multiple_of: int = 1,
            quantize_dropout_cutoff_index: int = 1,
            kmeans_init: bool = False,
            sync_kmeans: bool = True,
            sync_codebook: bool = False,
            quantize_dropout: bool = False,
    ):
        super().__init__()
        self.rq = ResidualVQ(
            dim=input_size,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            decay=ema_decay,
            commitment_weight=commitment_weight,
            quantize_dropout_multiple_of=quantize_dropout_multiple_of,
            threshold_ema_dead_code=2,
            quantize_dropout=quantize_dropout,
            quantize_dropout_cutoff_index=quantize_dropout_cutoff_index,
            kmeans_init=kmeans_init,
            sync_kmeans=sync_kmeans,
            sync_codebook=sync_codebook,
        )
        self.code_dim = input_size

    def forward(
            self,
            x,
    ):

        x, indices, commit_loss = self.rq(x)

        return x, indices, commit_loss

    def output_size(self):
        return self.code_dim
