import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, List, Tuple
from funcodec.modules.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
)
import logging


class Conv2dSubsampling2(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, hdim, odim, dropout_rate, patch_size=(16, 32), pos_enc=None):
        """Construct an Conv2dSubsampling2 object."""
        super(Conv2dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, hdim, 4, 2, padding=1, padding_mode="reflect"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hdim, hdim, 3, 1, padding=1, padding_mode="reflect"),
            torch.nn.ReLU(),
        )
        self.patch_fn = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        from funcodec.modules.embedding import PositionalEncoding
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hdim * patch_size[0] * patch_size[1], odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)    # (b, c, floor(t/2), floor(f/2))
        patches = self.patch_fn(x)  # (b, c, n)
        x = self.out(patches.transpose(1, 2))

        return x


class ViTEncoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            vit_input_layer: str = "conv2d2",
            sequence_model_type: str = "conformer",
            path_size: Union[int, List[int], Tuple[int]] = (16, 32),
            **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        output_size = kwargs.get("output_size", 512)
        positional_dropout_rate = kwargs.get("positional_dropout_rate", 0.1)

        rel_pos_type = kwargs.get("rel_pos_type", "legacy")
        pos_enc_layer_type = kwargs.get("pos_enc_layer_type", "rel_pos")
        selfattention_layer_type = kwargs.get("selfattention_layer_type", "rel_selfattn")

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if selfattention_layer_type == "rel_selfattn":
                selfattention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert selfattention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert selfattention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if vit_input_layer == "conv2d2":
            self.vit_input_layer = Conv2dSubsampling2(
                idim=input_size,
                hdim=input_size * 4,
                odim=output_size,
                dropout_rate=kwargs.get("dropout_rate", 0.1),
                patch_size=path_size,
                pos_enc=pos_enc_class(output_size, positional_dropout_rate),
            )
        else:
            raise TypeError(f"Unknown input layer type {vit_input_layer}.")

        self.sequence_model = None
        if sequence_model_type == "conformer":
            from funcodec.models.encoder.conformer_encoder import ConformerEncoder
            self.sequence_model = ConformerEncoder(
                input_size=output_size,
                **kwargs,
            )