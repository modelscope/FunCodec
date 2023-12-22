#!/usr/bin/env python3
# Copyright FunCodec (https://github.com/alibaba-damo-academy/FunCodec). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import argparse
import logging
import os
import sys
import math
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import kaldiio
import librosa
import numpy as np
import torch
import torchaudio
from einops import rearrange

from funcodec.utils.cli_utils import get_commandline_args
from funcodec.tasks.gan_speech_codec import GANSpeechCodecTask
from funcodec.torch_utils.device_funcs import to_device
from funcodec.torch_utils.set_all_random_seed import set_all_random_seed
from funcodec.utils import config_argparse
from funcodec.utils.types import str2bool
from funcodec.utils.types import str2triple_str
from funcodec.utils.types import str_or_none
from funcodec.utils.misc import statistic_model_parameters
import json
import torch.nn as nn
from thop import profile
from funcodec.torch_utils.model_summary import tree_layer_info
from funcodec.utils.hinter import hint_once


class Speech2Token(nn.Module):
    """Speech2Token class

    Examples:
        >>> import soundfile
        >>> speech2token = Speech2Token("config.yml", "model.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2token(audio)
        [(token_id, token_embed, recon_speech), ...]

    """

    def __init__(
            self,
            config_file: Union[Path, str] = None,
            model_file: Union[Path, str] = None,
            device: str = "cpu",
            batch_size: int = 1,
            dtype: str = "float32",
            streaming: bool = False,
            sampling_rate: int = 24_000,
            bit_width: int = 24_000,
    ):
        super().__init__()

        # 1. Build model
        import yaml
        with open(config_file, "rt", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        model, model_args = GANSpeechCodecTask.build_model_from_file(
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

    @torch.no_grad()
    def __call__(
            self,
            speech: Union[torch.Tensor, np.ndarray],
            ppg: Optional[Union[torch.Tensor, np.ndarray]] = None,
            need_recon: bool = True,
            bit_width: int = None,
            use_scale: bool = True,
            run_mod: str = "inference",
    ):
        """Inference

        Args:
            speech: Input speech data
        Returns:
            token_id, token_emb, recon_speech

        """
        self.model.eval()
        if isinstance(speech, np.ndarray):
            speech = torch.from_numpy(speech)
        if isinstance(ppg, np.ndarray):
            ppg = torch.from_numpy(ppg)
        speech = speech.to(self.device)
        batch = [speech,]
        if ppg is not None:
            ppg = ppg.to(self.device)
            batch = [speech, ppg]
        if run_mod == "inference":
            ret_dict = self.model.inference(*batch, need_recon=need_recon, bit_width=bit_width, use_scale=use_scale)
        elif run_mod == "encode":
            ret_dict = self.model.inference_encoding(*batch, need_recon=False, bit_width=bit_width)
        elif run_mod == "decode_emb":
            ret_dict = self.model.inference_decoding_emb(*batch)
        else:
            bit_per_quant = (self.model.quantizer.sampling_rate // self.model.quantizer.encoder_hop_length) * int(math.log2(self.model.quantizer.codebook_size))
            nq = None
            if bit_width is not None:
                nq = int(max(bit_width // bit_per_quant, 1))
            batch[0] = batch[0][:, :, :nq]
            hint_once(f"use {batch[0].shape[-1]} quantizers.", "infer_quantizer_num")
            ret_dict = self.model.inference_decoding(*batch)
        results = (
            ret_dict["code_indices"],
            ret_dict["code_embeddings"],
            ret_dict["recon_speech"],
            ret_dict["sub_quants"],
        )
        return results

    @staticmethod
    def from_pretrained(
            model_tag: Optional[str] = None,
            **kwargs: Optional[Any],
    ):
        """Build Speech2Token instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models. Currently, not used.

        Returns:
            Speech2Token: Speech2Token instance.

        """
        return Speech2Token(**kwargs)


def save_audio(wav: torch.Tensor, path: Union[Path, str],
               sample_rate: int, rescale: bool = False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    torchaudio.save(path, wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)


def inference_modelscope(
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
        sampling_rate: int = 16_000,
        bit_width: int = 8_000,
        param_dict: Optional[dict] = None,
        use_scale: Optional[bool] = True,
        **kwargs,
):
    # param_dict is used by modelscope, kwargs is used by argparser
    if param_dict is not None:
        kwargs.update(param_dict)

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
    logging.info("param_dict: {}".format(param_dict))

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
        sampling_rate = sampling_rate,
        bit_width = bit_width,
    )
    logging.info("model_kwargs: {}".format(model_kwargs))
    my_model = Speech2Token.from_pretrained(
        model_tag=model_tag,
        **model_kwargs,
    )
    my_model.model.eval()
    my_model.already_stat_flops = False

    def _forward(
            data_path_and_name_and_type: Sequence[Tuple[str, str, str]] = None,
            raw_inputs: Union[np.ndarray, torch.Tensor] = None,
            output_dir_v2: Optional[str] = None,
            param_dict: Optional[dict] = None,
    ):
        logging.info("param_dict: {}".format(param_dict))
        if param_dict is not None:
            kwargs.update(param_dict)
        if data_path_and_name_and_type is None and raw_inputs is not None:
            uttid = "utt"
            if isinstance(raw_inputs, str):
                uttid = os.path.basename(raw_inputs).rsplit(".")[0]
                raw_inputs, sr = librosa.load(raw_inputs, sr=sampling_rate)
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_dict=dict(
                speech=raw_inputs[np.newaxis, :],
                speech_lengths=torch.tensor([raw_inputs.shape[0]], dtype=torch.int64)
            )
            loader = [([uttid], data_dict)]
        else:
            # 3. Build data-iterator
            loader = GANSpeechCodecTask.build_streaming_iterator(
                data_path_and_name_and_type,
                dtype=dtype,
                batch_size=batch_size,
                key_file=key_file,
                num_workers=num_workers,
                preprocess_fn=None,
                collate_fn=GANSpeechCodecTask.build_collate_fn(argparse.Namespace(
                    float_pad_value=0.0,
                    int_pad_value=0,
                    pad_mode="wrap",
                ), False),
                allow_variable_data_keys=allow_variable_data_keys,
                inference=True,
            )

        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        result_list = []
        should_resample = False
        if "file_sampling_rate" in kwargs and kwargs["file_sampling_rate"] != sampling_rate:
            logging.info(f"Resample from {kwargs['file_sampling_rate']} to {sampling_rate}.")
            should_resample = True

        indices_writer = None
        if "need_indices" in kwargs and kwargs["need_indices"]:
            if "indices_save_type" in kwargs and kwargs["indices_save_type"] == "ark":
                outfile_path = os.path.join(output_path, "indices")
                indices_writer = kaldiio.WriteHelper("ark,scp,f:{}.ark,{}.scp".format(outfile_path, outfile_path))
            else:
                indices_writer = open(os.path.join(output_path, "codecs.txt"), "wt")

        sub_quants_writer = None
        if "need_sub_quants" in kwargs and kwargs["need_sub_quants"]:
            outfile_path = os.path.join(output_path, "codec_emb")
            sub_quants_writer = kaldiio.WriteHelper("ark,scp,f:{}.ark,{}.scp".format(outfile_path, outfile_path))

        def write_indices(_key, _indices, batch_id=0, length=None):
            if indices_writer is None:
                return
            if "indices_save_type" in kwargs and kwargs["indices_save_type"] == "ark":
                to_write = [x[:, batch_id, :length].cpu().float().numpy().T for x in _indices]
                to_write = np.concatenate(to_write, axis=0)
                indices_writer(_key, to_write)
            else:
                # n_frame x n_q x B x T, n_frame is always 1
                to_write = [x[:, batch_id, :length].cpu().numpy().tolist() for x in _indices]
                json_str = json.dumps(to_write)
                indices_writer.write(_key + " " + json_str + "\n")

        def write_sub_quants(_key, _sub_quants, batch_id=0, length=None):
            if sub_quants_writer is None:
                return
            # n_q x B x D x T
            to_write = torch.cat(_sub_quants, dim=-1)
            # T x n_q x D
            to_write = to_write.permute(1, 3, 0, 2)[batch_id][:length]
            to_write = rearrange(to_write, "t ... -> t (...)")
            to_write = to_write.cpu().numpy()

            sub_quants_writer(_key, to_write)

        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            if should_resample:
                batch["speech"] = torchaudio.functional.resample(
                    batch["speech"],
                    orig_freq=kwargs["file_sampling_rate"],
                    new_freq=sampling_rate)

            speech_length = batch.pop("speech_lengths")
            if "ppg_lengths" in batch:
                ppg_length = batch.pop("ppg_lengths")

            if 'stat_flops' in kwargs and kwargs["stat_flops"] and not my_model.already_stat_flops:
                rand_speech = torch.randn(1, sampling_rate, device=device, dtype=torch.float32)
                if "ppg" in batch:
                    rand_ppg = torch.randn(1, 100, batch["ppg"].shape[-1],
                                           device=device, dtype=torch.float32)
                    model_inputs = (rand_speech, rand_ppg, True, bit_width, use_scale, "inference")
                else:
                    model_inputs = (rand_speech, None, True, bit_width, use_scale, "inference")
                # macs, params = profile(my_model, inputs=model_inputs, verbose=False)
                # macs, params = clever_format([macs, params], "%.2f")
                # logging.info(f"Model parameters: {params}, model flops: {macs}.")
                macs, params, layer_info = profile(my_model, inputs=model_inputs, verbose=False, ret_layer_info=True)
                layer_info = tree_layer_info(macs, params, layer_info, 0)
                logging.info(f"Model layer info: \n{layer_info}")
                my_model.already_stat_flops = True

            run_mod = kwargs.get("run_mod", "inference")
            token_id, token_emb, recon_speech, sub_quants = my_model(
                **batch, need_recon=True,
                bit_width=param_dict["bit_width"] if param_dict is not None and "bit_width" in param_dict else bit_width,
                use_scale=use_scale,
                run_mod=run_mod
            )

            if should_resample and recon_speech is not None:
                recon_speech = torchaudio.functional.resample(
                    recon_speech,
                    orig_freq=sampling_rate,
                    new_freq=kwargs["file_sampling_rate"])

            for i, key in enumerate(keys):
                recon_wav = None
                if run_mod in ["decode", "decode_emb"]:
                    codec_len = speech_length[i]
                    ilen = codec_len * my_model.model.quantizer.encoder_hop_length
                else:
                    ilen = speech_length[i]
                    codec_len = torch.ceil(ilen / my_model.model.quantizer.encoder_hop_length).int().item()
                if recon_speech is not None:
                    recon_wav = recon_speech[i].cpu()[:, :ilen]
                item = {"key": key, "value": recon_wav}
                if output_path is not None:
                    if recon_wav is not None:
                        save_audio(recon_wav, os.path.join(output_path, key+".wav" if not key.endswith(".wav") else key), rescale=True,
                                   sample_rate=kwargs["file_sampling_rate"] if should_resample else sampling_rate)
                    if token_id is not None:
                        write_indices(key, token_id, batch_id=i, length=codec_len)
                    if sub_quants is not None:
                        write_sub_quants(key, sub_quants, batch_id=i, length=codec_len)
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
        sampling_rate: int = 24_000,
        bit_width: int = 24_000,
        use_scale: bool = True,
        **kwargs,
):
    inference_pipeline = inference_modelscope(
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
        sampling_rate=sampling_rate,
        bit_width=bit_width,
        use_scale=use_scale,
        **kwargs,
    )

    return inference_pipeline(data_path_and_name_and_type, raw_inputs=None)


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Speech Tokenizer",
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
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
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
        "--sampling_rate",
        type=int,
        default=24_000,
        help="The sampling rate"
    )
    parser.add_argument(
        "--file_sampling_rate",
        type=int,
        default=None,
        help="The sampling rate"
    )
    parser.add_argument(
        "--bit_width",
        type=int,
        default=16_000,
        help="The bit width for quantized code."
    )
    parser.add_argument(
        "--use_scale",
        type=str2bool,
        default=True,
        help="Whether use scale for decoding."
    )
    group.add_argument(
        "--need_indices",
        type=str2bool,
        help="whether to dump code index",
    )
    group.add_argument(
        "--indices_save_type",
        type=str,
        default="text",
        help="whether to dump code index",
    )
    group.add_argument(
        "--need_sub_quants",
        type=str2bool,
        help="whether to dump sub quantized",
    )
    group.add_argument(
        "--run_mod",
        type=str,
        choices=["inference", "encode", "decode", "decode_emb"],
        default="inference",
        help="run mode",
    )
    group.add_argument(
        "--stat_flops",
        type=str2bool,
        default=False,
        help="whether to statistic flops",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    if args.file_sampling_rate is None:
        args.file_sampling_rate = args.sampling_rate
    kwargs = vars(args)
    kwargs.pop("config", None)
    if args.output_dir is None:
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
