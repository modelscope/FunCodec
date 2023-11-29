import argparse
import logging
import os
from pathlib import Path
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import yaml

from funcodec.datasets.collate_fn import CommonCollateFn
from funcodec.datasets.preprocessor import Text2AudioPreprocessor
from funcodec.models.encoder.abs_encoder import AbsEncoder
from funcodec.models.encoder.rnn_encoder import RNNEncoder
from funcodec.models.encoder.transformer_encoder import TransformerEncoder
from funcodec.models.encoder.conformer_encoder import ConformerEncoder
from funcodec.tasks.abs_task import AbsTask
from funcodec.models.audio_generation.laura_model import LauraGenModel
from funcodec.models.decoder.seanet_decoder import SEANetDecoder, SEANetDecoder2d
from funcodec.torch_utils.initialize import initialize
from funcodec.train.abs_espnet_model import AbsESPnetModel
from funcodec.train.class_choices import ClassChoices
from funcodec.train.trainer import Trainer
from funcodec.utils.types import int_or_none
from funcodec.utils.types import str2bool
from funcodec.utils.types import str_or_none

model_choices = ClassChoices(
    "model",
    classes=dict(
        laura_gen_model=LauraGenModel,
    ),
    type_check=torch.nn.Module,
    default="laura_gen_model",
)
text_encoder_choices = ClassChoices(
    name="text_encoder",
    classes=dict(
        transformer=TransformerEncoder,
        conformer=ConformerEncoder,
    ),
    type_check=torch.nn.Module,
    optional=True,
    default="transformer",
)
codec_encoder_choices = ClassChoices(
    "codec_encoder",
    classes=dict(
        transformer=TransformerEncoder,
        conformer=ConformerEncoder,
        rnn=RNNEncoder,
    ),
    type_check=torch.nn.Module,
    optional=True,
    default="transformer",
)


class Text2AudioGenTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --model and --model_conf
        model_choices,
        # --text_encoder and --text_encoder_conf
        text_encoder_choices,
        # --codec_encoder and --codec_encoder_conf
        codec_encoder_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        # required = parser.get_default("required")
        # required += ["token_list"]

        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of seq dimension of the feature",
        )
        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="The path to token list file",
        )
        group.add_argument(
            "--token_type",
            type=str_or_none,
            default=None,
            help="The token type",
        )
        group.add_argument(
            "--g2p_type",
            type=str_or_none,
            default=None,
            help="The g2p type",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--audio_max_duration",
            type=int,
            default=30,
            help="The max duration of audio outputs",
        )
        group.add_argument(
            "--codec_token_rate",
            type=int,
            default=25,
            help="The max duration of audio outputs",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
            cls, args: argparse.Namespace, train: bool, raw_sequence=()
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1, raw_sequence=raw_sequence)

    @classmethod
    def build_preprocess_fn(
            cls, args: argparse.Namespace, train: bool,
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        if args.use_preprocessor:
            retval = Text2AudioPreprocessor(
                train=train,
                audio_max_duration=args.audio_max_duration,
                codec_token_rate=args.codec_token_rate,
                token_list=args.token_list,
                token_type=args.token_type,
                g2p_type=args.g2p_type if hasattr(args, "g2p_type") else None,
            )
        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
            cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("text", "codec")
        else:
            # generation mode
            retval = ("text",)
        return retval

    @classmethod
    def optional_data_names(
            cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ()
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace):
        input_size = args.input_size

        # 1. Text Encoder
        if args.text_encoder is not None:
            text_encoder_class = text_encoder_choices.get_class(args.text_encoder)
            text_encoder = text_encoder_class(input_size=input_size, **args.text_encoder_conf)
        else:
            text_encoder = None

        # 2. Codec Encoder
        if args.codec_encoder is not None:
            codec_encoder_class = codec_encoder_choices.get_class(args.codec_encoder)
            codec_encoder = codec_encoder_class(
                input_size=args.model_conf["codec_conf"]["codebook_dim"],
                **args.codec_encoder_conf
            )
        else:
            codec_encoder = None

        # 3. Build model
        token_list = []
        if args.token_list is not None:
            if isinstance(args.token_list, list):
                token_list = args.token_list
            elif os.path.exists(args.token_list):
                for line in open(args.token_list, "rt"):
                    token = line.strip()
                    token_list.append(token)
            else:
                raise TypeError("If token_list is not None, it must be list or str.")
        model_class = model_choices.get_class(args.model)
        model = model_class(
            input_size=input_size,
            vocab_size=len(token_list),
            token_list=token_list,
            text_encoder=text_encoder,
            codec_encoder=codec_encoder,
            **args.model_conf,
        )

        # 10. Initialize
        if args.init is not None:
            initialize(model, args.init)

        return model
