from typing import List
from typing import Optional
from typing import Tuple
import torch
from typeguard import check_argument_types
from funcodec.models.encoder.abs_encoder import AbsEncoder
from funcodec.modules.attention import MultiHeadedAttention
from funcodec.modules.embedding import PositionalEncoding
from funcodec.modules.layer_norm import LayerNorm
from funcodec.modules.multi_layer_conv import Conv1dLinear
from funcodec.modules.multi_layer_conv import MultiLayeredConv1d
from funcodec.modules.nets_utils import make_pad_mask
from funcodec.modules.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from funcodec.modules.repeat import repeat
from funcodec.modules.subsampling import Conv2dSubsampling
from funcodec.modules.subsampling import Conv2dSubsampling2
from funcodec.modules.subsampling import Conv2dSubsampling6
from funcodec.modules.subsampling import Conv2dSubsampling8
from funcodec.modules.subsampling import TooShortUttError
from funcodec.modules.subsampling import check_short_utt
from funcodec.models.encoder.transformer_encoder import EncoderLayer


class TransformerEncoder(torch.nn.Module):
    """Transformer encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
            self,
            input_size: int,
            output_size: int = 512,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 4,
            dropout_rate: float = 0.0,
            positional_dropout_rate: float = 0.0,
            attention_dropout_rate: float = 0.0,
            input_layer: Optional[str] = None,
            pos_enc_class=PositionalEncoding,
            normalize_before: bool = True,
            concat_after: bool = False,
            positionwise_layer_type: str = "linear",
            positionwise_conv_kernel_size: int = 1,
            padding_idx: int = -1,
            causal_mode: str = "None",
            skip: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        self.causal_mode = causal_mode
        self.skip = skip

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(input_size, output_size, dropout_rate)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer is None:
            if input_size == output_size:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                MultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs_pad: torch.Tensor,
            ilens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, C, L)
            ilens: input length (B)
        Returns:
            position embedded tensor and mask
        """
        xs_pad = xs_pad.permute(0, 2, 1)
        residual = xs_pad
        if ilens is not None:
            masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        else:
            masks = torch.ones(xs_pad.shape[0], 1, xs_pad.shape[1],
                               dtype=torch.bool, device=xs_pad.device)
        if self.causal_mode == "None":
            pass
        elif self.causal_mode == "causal":
            tt = xs_pad.shape[1]
            pos_idx = torch.arange(tt)
            causal_mask = torch.less_equal(pos_idx.unsqueeze(0), pos_idx.unsqueeze(1))
            causal_mask = causal_mask.unsqueeze(0)
            masks = masks * causal_mask

        if self.embed is None:
            xs_pad = xs_pad
        elif (
                isinstance(self.embed, Conv2dSubsampling)
                or isinstance(self.embed, Conv2dSubsampling2)
                or isinstance(self.embed, Conv2dSubsampling6)
                or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        xs_pad, masks = self.encoders(xs_pad, masks)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        # olens = masks.squeeze(1).sum(1)
        if self.skip:
            xs_pad = xs_pad + residual

        return xs_pad.permute(0, 2, 1)
