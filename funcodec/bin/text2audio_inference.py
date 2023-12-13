#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import kaldiio
import numpy as np
import torch
import torchaudio
from funcodec.utils.cli_utils import get_commandline_args
from funcodec.tasks.text2audio_generation import Text2AudioGenTask
from funcodec.torch_utils.set_all_random_seed import set_all_random_seed
from funcodec.utils import config_argparse
from funcodec.utils.types import str2bool
from funcodec.utils.types import str2triple_str
from funcodec.utils.types import str_or_none
from funcodec.utils.types import int_or_float_or_bool
from funcodec.utils.misc import statistic_model_parameters
import torch.nn as nn
import librosa
from torch.nn import functional as F


class Text2Audio(nn.Module):
    """Text2Audio class

    Examples:
        >>> text2audio = Text2Audio("config.yml", "model.pth")
        >>> text = input("Input a prompt in english:")
        >>> text2audio(text)
        [(token_id, token_embed, recon_speech), ...]

    """

    def __init__(
            self,
            config_file: Union[Path, str] = None,
            model_file: Union[Path, str] = None,
            device: str = "cpu",
            dtype: str = "float32",
            **kwargs
    ):
        super().__init__()

        # 1. Build model
        model, model_args = Text2AudioGenTask.build_model_from_file(
            config_file=config_file,
            model_file=model_file,
            device=device
        )
        logging.info("model: {}".format(model))
        logging.info("model parameter number: {}".format(statistic_model_parameters(model)))
        logging.info("model arguments: {}".format(model_args))
        model.to(dtype=getattr(torch, dtype)).eval()

        self.model = model
        self.model_args = model_args
        self.device = device
        self.dtype = dtype
        self.already_stat_flops = False
        text_emb_model = kwargs["text_emb_model"]
        self.beam_size = kwargs["beam_size"]
        self.sampling = kwargs["sampling"]
        self.continual = kwargs["continual"]
        self.tokenize_to_phone = kwargs.get("tokenize_to_phone", False)
        self.exclude_prompt = kwargs.get("exclude_prompt", True)
        if self.tokenize_to_phone:
            from funcodec.text.phoneme_tokenizer import G2p_en
            self.phoneme_tokenizer = G2p_en(no_space=True)

        if not hasattr(self.model, "vocab_size") or self.model.vocab_size == 0:
            # 1. Build text embedding model
            self.text_emb_model = self.build_text_emb_model(text_emb_model)
        else:
            self.text_emb_model = self.tokenize_text

        # 2. Build codec model
        from funcodec.bin.codec_inference import Speech2Token
        codec_kwargs = dict(
            config_file=kwargs["codec_config_file"],
            model_file=kwargs["codec_model_file"],
            device=device,
        )
        self.codec_model = Speech2Token.from_pretrained(
            model_tag=None,
            **codec_kwargs,
        )

    def tokenize_text(self, text: str):
        if self.tokenize_to_phone:
            phoneme_list = self.phoneme_tokenizer(text)
        else:
            phoneme_list = text.strip().split(" ")
        logging.info(" ".join(phoneme_list))
        token_ids = []
        for one in phoneme_list:
            if one in self.model.token_list:
                token_ids.append(self.model.token_list.index(one))
        logging.info(" ".join([str(x) for x in token_ids]))
        token_idx = torch.Tensor(token_ids).long().to(self.device)
        text_emb = self.model.token_embedding(token_idx)

        return text_emb.unsqueeze(0), torch.Tensor([text_emb.shape[0]]).long().to(self.device)

    def build_text_emb_model(self, model_path: str):
        emb_type = "enc"
        if ":" in model_path:
            model_path, emb_type = model_path.rsplit(":", maxsplit=1)
        logging.info("loading text_emb model...")
        from transformers import T5Tokenizer, T5Model
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5Model.from_pretrained(model_path)
        model = model.to(self.device)
        logging.info("done.")

        def _forward(text: str):
            inputs = tokenizer(text, return_tensors="pt")
            for key in inputs.keys():
                inputs[key] = inputs[key].to(self.device)
            with torch.no_grad():
                if emb_type == "enc":
                    outputs = model.encoder(inputs.input_ids).last_hidden_state
                else:
                    outputs = model.shared(inputs.input_ids)
            out_lens = inputs["attention_mask"].sum(dim=1)
            return outputs, out_lens

        return _forward

    @torch.no_grad()
    def __call__(
            self,
            text: str,
            prompt_text: str = None,
            prompt_audio: np.ndarray = None,
    ):
        """Inference

        Args:
            text: Input text data
            prompt_text: Prompt text for zero-shot adaption
            prompt_audio: Prompt audio for zero-shot adaption
        Returns:
            generation audios
        """
        self.model.eval()
        continual_mode = self.continual and prompt_text is not None and prompt_audio is not None
        if continual_mode:
            text = " ".join([prompt_text, text]).strip()
            codec = self.codec_model(prompt_audio, run_mod="encode")[0][0].squeeze(1).transpose(0,1)
            continual = codec[:, :self.model.predict_nq].tolist()
            continual_length = len(continual) if self.exclude_prompt else 0
        else:
            continual = None
            continual_length = None

        # 0. extract text embeddings
        text_emb, text_emb_lens = self.text_emb_model(text)

        # 1. encode text
        text_outs, text_out_lens = self.model.encode(text_emb, text_emb_lens)

        # 2. decode first codec group
        decoded_codec = self.model.decode_codec(
            text_outs,
            text_out_lens,
            max_length=30 * 25,
            sampling=self.sampling,
            beam_size=self.beam_size,
            continual=continual
        )

        _, _, gen_speech_only_lm, _ = self.codec_model(
            decoded_codec[:, continual_length:],
            bit_width=None,
            run_mod="decode"
        )

        # 3. predict embeddings
        gen_speech = self.model.syn_audio(
            decoded_codec, text_outs, text_out_lens, self.codec_model,
            continual_length=continual_length,
        )

        ret_val = dict(
            gen=gen_speech,
            gen_only_lm=gen_speech_only_lm,
        )

        return ret_val, decoded_codec

    @staticmethod
    def from_pretrained(
            model_tag: Optional[str] = None,
            **kwargs: Optional[Any],
    ):
        """Build Speech2Xvector instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            Speech2Xvector: Speech2Xvector instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return Text2Audio(**kwargs)


def save_audio(wav: torch.Tensor, path: Union[Path, str],
               sample_rate: int, rescale: bool = False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1) * 0.6
    else:
        wav = wav.clamp(-limit, limit)
    torchaudio.save(path, wav.cpu(), sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)


def inference_func(
        output_dir: Optional[str] = None,
        batch_size: int = 1,
        dtype: str = "float32",
        ngpu: int = 1,
        seed: int = 0,
        num_workers: int = 0,
        log_level: Union[int, str] = "INFO",
        key_file: Optional[str] = None,
        config_file: Optional[str] = "config.yaml",
        model_file: Optional[str] = "model.pth",
        model_tag: Optional[str] = None,
        allow_variable_data_keys: bool = True,
        streaming: bool = False,
        **kwargs,
):
    if batch_size > 1:
        logging.info(f"batch_size = {batch_size}")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build model
    model_kwargs = dict(
        config_file=config_file,
        model_file=model_file,
        device=device,
        dtype=dtype,
        streaming=streaming,
        **kwargs
    )
    logging.info("model_kwargs: {}".format(model_kwargs))
    my_model = Text2Audio.from_pretrained(
        model_tag=model_tag,
        **model_kwargs,
    )
    my_model.model.eval()

    def _forward(
            data_path_and_name_and_type: Sequence[Tuple[str, str, str]] = None,
            raw_inputs: Union[Tuple[str], Tuple[str, str, str]] = None,
            output_dir_v2: Optional[str] = None,
            param_dict: Optional[dict] = None,
    ):
        logging.info("param_dict: {}".format(param_dict))
        if data_path_and_name_and_type is None and raw_inputs is not None:
            # add additional parenthesis to keep the same data format as streaming_iterator
            data_dict = dict(
                text=[raw_inputs[0]]
            )
            if len(raw_inputs) == 3:
                data_dict["prompt_text"] = [raw_inputs[1]]
                if isinstance(raw_inputs[2], str):
                    data_dict["prompt_audio"] = [librosa.load(
                        raw_inputs[2],
                        sr=my_model.codec_model.model.quantizer.sampling_rate,
                        mono=True,
                        dtype=np.float32
                    )[0][np.newaxis, :]]
                else:
                    data_dict["prompt_audio"] = [raw_inputs[2].squeeze()[None, :]]
            loader = [(["utt1"], data_dict)]
        else:
            loader = Text2AudioGenTask.build_streaming_iterator(
                data_path_and_name_and_type,
                dtype=dtype,
                batch_size=batch_size,
                key_file=key_file,
                num_workers=num_workers,
                preprocess_fn=None,
                collate_fn=Text2AudioGenTask.build_collate_fn(my_model.model_args, False,
                                                              raw_sequence=("text", "prompt_text")),
                allow_variable_data_keys=allow_variable_data_keys,
                inference=True,
            )

        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        result_list = []

        for keys, data in loader:
            key = keys[0]
            logging.info(f"generating {key}")
            model_inputs = [data["text"][0]]
            for input_key in ["prompt_text", "prompt_audio"]:
                if input_key in data:
                    model_inputs.append(data[input_key][0])

            ret_val, _ = my_model(*model_inputs)
            item = {"key": key, "value": ret_val}
            if output_path is not None:
                for suffix, wave in ret_val.items():
                    file_name = key.replace(".wav", "") + "_" + suffix + ".wav"
                    save_path = os.path.join(output_path, file_name)
                    save_audio(
                        wave[0],
                        save_path,
                        rescale=True,
                        sample_rate=my_model.codec_model.model.quantizer.sampling_rate
                    )
            else:
                result_list.append(item)

        return result_list

    return _forward


def inference(
        output_dir: Optional[str],
        batch_size: int,
        dtype: str,
        ngpu: int,
        seed: int,
        num_workers: int,
        log_level: Union[int, str],
        data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
        key_file: Optional[str],
        config_file: Optional[str],
        model_file: Optional[str],
        model_tag: Optional[str],
        allow_variable_data_keys: bool = True,
        streaming: bool = False,
        **kwargs,
):
    inference_pipeline = inference_func(
        output_dir=output_dir,
        batch_size=batch_size,
        dtype=dtype,
        ngpu=ngpu,
        seed=seed,
        num_workers=num_workers,
        log_level=log_level,
        key_file=key_file,
        config_file=config_file,
        model_file=model_file,
        model_tag=model_tag,
        allow_variable_data_keys=allow_variable_data_keys,
        streaming=streaming,
        **kwargs,
    )

    return inference_pipeline(data_path_and_name_and_type, raw_inputs=kwargs.get("raw_inputs", None))


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Text to audio generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument(
        "--gpuid_list",
        type=str,
        default="",
        help="The visible gpus",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="The number of workers used for DataLoader",
    )
    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=False,
        action="append",
    )
    group.add_argument(
        "--raw_inputs",
        type=str,
        required=False,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--mode",
        type=str,
        default="inference mode",
        help="llm_codec",
    )
    group.add_argument(
        "--config_file",
        type=str,
        help="path to configuration file",
    )
    group.add_argument(
        "--model_file",
        type=str,
        help="path to model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
             "*_file will be overwritten",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="The beam size for inference",
    )
    group.add_argument(
        "--text_emb_model",
        type=str,
        default="./exp/t5-base",
        help="The path to text embedding extraction model",
    )
    group.add_argument(
        "--sampling",
        type=int_or_float_or_bool,
        default='true',
        help="Sampling method",
    )
    group.add_argument(
        "--codec_config_file",
        type=str,
        default=None,
        help="Path to config file of codec model",
    )
    group.add_argument(
        "--codec_model_file",
        type=str,
        default=None,
        help="Path to parameter file of codec model",
    )
    group.add_argument(
        "--continual",
        type=int,
        default=0,
        help="Path to parameter file of codec model",
    )
    group.add_argument(
        "--tokenize_to_phone",
        type=str2bool,
        default=False,
        help="whether tokenize the input text into phoneme sequence."
    )
    group.add_argument(
        "--exclude_prompt",
        type=str2bool,
        default=True,
        help="whether tokenize the input text into phoneme sequence."
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    if args.output_dir is None or "." not in args.output_dir:
        jobid, n_gpu = 1, 1
        gpuid = args.gpuid_list.split(",")[jobid-1]
    else:
        jobid = int(args.output_dir.split(".")[-1])
        n_gpu = len(args.gpuid_list.split(","))
        gpuid = args.gpuid_list.split(",")[(jobid - 1) % n_gpu]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    if torch.__version__ >= "1.10":
        torch.cuda.set_device(int(gpuid))
    inference(**kwargs)


if __name__ == "__main__":
    main()
