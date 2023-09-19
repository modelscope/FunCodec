# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based text-to-speech task."""

import argparse
import logging

from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from funcodec.torch_utils.model_summary import model_summary
import numpy as np
import torch

from typeguard import check_argument_types
from typeguard import check_return_type

from funcodec.train.abs_gan_espnet_model import AbsGANESPnetModel
from funcodec.models.codec_basic import Encodec
from funcodec.models.codec_freq import FreqCodec
from funcodec.models.codec_semantic_aug import CodecSemanticAug
from funcodec.models.encoder.seanet_encoder import SEANetEncoder, SEANetEncoder2d
from funcodec.models.decoder.seanet_decoder import SEANetDecoder, SEANetDecoder2d
from funcodec.models.quantizer.identity_quantizer import IdentityQuantizer
from funcodec.models.quantizer.costume_quantizer import CostumeQuantizer
from funcodec.models.discriminator.multiple_discriminator import MultipleDiscriminator
from funcodec.layers.abs_normalize import AbsNormalize
from funcodec.layers.global_mvn import GlobalMVN
from funcodec.layers.utterance_mvn import UtteranceMVN
from funcodec.tasks.abs_task import AbsTask
from funcodec.tasks.abs_task import optim_classes
from funcodec.train.class_choices import ClassChoices
from funcodec.datasets.collate_fn import CommonCollateFn
from funcodec.train.gan_trainer import GANTrainer
from funcodec.datasets.preprocessor import CodecPreprocessor
from funcodec.utils.types import float_or_none
from funcodec.utils.types import int_or_none
from funcodec.utils.types import str2bool
from funcodec.utils.types import str_or_none
from funcodec.models.frontend.abs_frontend import AbsFrontend
from funcodec.models.frontend.default import DefaultFrontend
from funcodec.models.frontend.fused import FusedFrontends
from funcodec.models.frontend.s3prl import S3prlFrontend
from funcodec.models.frontend.wav_frontend import WavFrontend
from funcodec.models.frontend.windowing import SlidingWindow

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
        fused=FusedFrontends,
        wav_frontend=WavFrontend,
    ),
    type_check=AbsFrontend,
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default=None,
    optional=True,
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        encodec_seanet_encoder=SEANetEncoder,
        encodec_seanet_encoder_2d=SEANetEncoder2d,
    ),
    type_check=torch.nn.Module,
    default="encodec_seanet_encoder",
)
quantizer_choices = ClassChoices(
    "quantizer",
    classes=dict(
        identity_quantizer=IdentityQuantizer,
        costume_quantizer=CostumeQuantizer,
    ),
    type_check=torch.nn.Module,
    default="costume_quantizer",
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        encodec_seanet_decoder=SEANetDecoder,
        encodec_seanet_decoder_2d=SEANetDecoder2d,
    ),
    type_check=torch.nn.Module,
    default="encodec_seanet_decoder",
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        encodec=Encodec,
        freq_codec=FreqCodec,
        codec_semantic_aug=CodecSemanticAug,
    ),
    type_check=AbsGANESPnetModel,
    default="encodec",
)
discriminator_choices = ClassChoices(
    "discriminator",
    classes=dict(
        multiple_disc=MultipleDiscriminator,
    ),
    type_check=torch.nn.Module,
    default="multiple_disc",
)


class GANSpeechCodecTask(AbsTask):
    """GAN-based speech tokenizer task."""

    # GAN requires two optimizers
    num_optimizers: int = 2

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --quantizer and --quantizer_conf
        quantizer_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --model and --model_conf
        model_choices,
        # --discriminator and --discriminator_conf
        discriminator_choices
    ]

    # Use GANTrainer instead of Trainer
    trainer = GANTrainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        assert check_argument_types()
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of dimension of inputs",
        )
        group.add_argument(
            "--cmvn_file",
            type=str_or_none,
            default=None,
            help="The cmvn file for features",
        )
        group.add_argument(
            "--disc_grad_clip",
            type=float,
            default=0.5,
            help="Gradient norm threshold to clip for discriminator",
        )
        group.add_argument(
            "--disc_grad_clip_type",
            type=float,
            default=2.0,
            help="The type of the used p-norm for gradient clip of discriminator. Can be inf",
        )
        group.add_argument(
            "--gen_train_interval",
            type=int,
            default=1,
            help="Update generator every `gen_train_interval` steps",
        )
        group.add_argument(
            "--disc_train_interval",
            type=int,
            default=1,
            help="Update discriminator every `gen_train_interval` steps",
        )
        group.add_argument(
            "--stat_flops",
            type=str2bool,
            default=False,
            help="whether to statistic flops."
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=False,
            help="Apply preprocessing to data or not",
        )
        parser.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value.",
        )
        parser.add_argument(
            "--speech_rms_normalize",
            type=str2bool,
            default=False,
            help="Whether to perform rms normalization.",
        )
        # speech_max_dur
        group.add_argument(
            "--speech_max_length",
            type=int,
            default=50000,
            help="The maximum duration of speech for training",
        )
        group.add_argument(
            "--sampling_rate",
            type=int,
            default=16_000,
            help="The sampling rate of input waveforms"
        )
        group.add_argument(
            "--valid_max_length",
            type=int,
            default=50000,
            help="The maximum duration of speech for valid"
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        return CommonCollateFn(
            float_pad_value=args.__dict__.get("float_pad_value", 0.0),
            int_pad_value=args.__dict__.get("int_pad_value", 0),
            pad_mode=args.__dict__.get("pad_mode", None)
        )

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            if train:
                max_length = args.speech_max_length
            else:
                max_length = args.valid_max_length
            retval = CodecPreprocessor(
                train=train,
                speech_max_length=max_length,
                speech_volume_normalize=args.speech_volume_normalize,
                speech_rms_normalize=args.speech_rms_normalize
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech",)
        else:
            # Inference mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("ppg",)
        else:
            # Inference mode
            retval = ("ppg",)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> AbsGANESPnetModel:
        assert check_argument_types()
        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            if args.frontend == 'wav_frontend':
                frontend = frontend_class(cmvn_file=args.cmvn_file, **args.frontend_conf)
            else:
                frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 3. Quantizer
        quantizer_class = quantizer_choices.get_class(args.quantizer)
        quantizer = quantizer_class(input_size=encoder.output_size(), **args.quantizer_conf)

        # 4. decoder
        decoder_class = decoder_choices.get_class(args.decoder)
        decoder = decoder_class(input_size=quantizer.output_size(), **args.decoder_conf)

        discriminator_class = discriminator_choices.get_class(args.discriminator)
        discriminator = discriminator_class(**args.discriminator_conf)

        model_class = model_choices.get_class(args.model)
        model = model_class(
            input_size=input_size,
            frontend=frontend,
            encoder=encoder,
            quantizer=quantizer,
            decoder=decoder,
            discriminator=discriminator,
            **args.model_conf,
        )

        if hasattr(args, "stat_flops") and args.stat_flops:
            sr = args.model_conf["target_sample_hz"]
            rand_speech = torch.randn(1, sr, device="cpu", dtype=torch.float32)
            model_inputs = (True, {"speech": rand_speech, "speech_lengths": torch.Tensor([sr]).long()})
            from thop import profile
            from thop import clever_format
            from funcodec.torch_utils.model_summary import tree_layer_info
            macs, params, layer_info = profile(model, inputs=model_inputs, verbose=False, ret_layer_info=True)
            layer_info = tree_layer_info(macs, params, layer_info, 0)
            macs, params = clever_format([macs, params], format="%.2f")
            logging.info(f"Flops: {macs}, Param: {params}, Model layer info: \n{layer_info}")

        assert check_return_type(model)
        return model

    @classmethod
    def build_optimizers(
        cls,
        args: argparse.Namespace,
        model: AbsGANESPnetModel,
    ) -> List[torch.optim.Optimizer]:
        # check
        assert hasattr(model, "generator")
        assert hasattr(model, "discriminator")

        # define generator optimizer
        optim_g_class = optim_classes.get(args.optim)
        if optim_g_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim}")
        if args.sharded_ddp:
            try:
                import fairscale
            except ImportError:
                raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
            optim_g = fairscale.optim.oss.OSS(
                params=model.tts.generator.parameters(),
                optim=optim_g_class,
                **args.optim_conf,
            )
        else:
            logging.info(f"Generator modules: {model_summary(model.generator)}.")
            optim_g = optim_g_class(
                model.generator.parameters(),
                **args.optim_conf,
            )
        optimizers = [optim_g]

        # define discriminator optimizer
        optim_d_class = optim_classes.get(args.optim2)
        if optim_d_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim2}")
        if args.sharded_ddp:
            try:
                import fairscale
            except ImportError:
                raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
            optim_d = fairscale.optim.oss.OSS(
                params=model.tts.discriminator.parameters(),
                optim=optim_d_class,
                **args.optim2_conf,
            )
        else:
            logging.info(f"Discriminator modules: {model_summary(model.discriminator)}.")
            optim_d = optim_d_class(
                model.discriminator.parameters(),
                **args.optim2_conf,
            )
        optimizers += [optim_d]

        return optimizers
