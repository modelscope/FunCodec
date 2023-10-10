import json
import logging
import os
import random
from functools import partial

import torch
import torch.distributed as dist
import torchaudio
from kaldiio import ReadHelper
from torch.utils.data import IterableDataset
import numpy as np

from funcodec.datasets.large_datasets.datapipes.batch import MaxTokenBucketizerIterDataPipe
from funcodec.datasets.large_datasets.datapipes.filter import FilterIterDataPipe
from funcodec.datasets.large_datasets.datapipes.map import MapperIterDataPipe
from funcodec.datasets.large_datasets.utils.clipping import clipping, clip_speech_fix_length
from funcodec.datasets.large_datasets.utils.filter import filter
from funcodec.datasets.large_datasets.utils.padding import padding
from funcodec.datasets.large_datasets.utils.tokenize import tokenize


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            parts = line.strip()
            lists.append(parts)
    return lists


class AudioDataset(IterableDataset):
    def __init__(self,
                 scp_lists, data_names, data_types,
                 frontend_conf=None, shuffle=True,
                 speed_perturb=None, mode="train"):
        self.scp_lists = scp_lists
        self.data_names = data_names
        self.data_types = data_types
        self.frontend_conf = frontend_conf
        self.shuffle = shuffle
        self.mode = mode
        self.epoch = -1
        self.rank = 0
        self.world_size = 1
        self.worker_id = 0
        self.num_workers = 1
        self.speed_perturb = speed_perturb
        if self.speed_perturb is not None:
            logging.info("Using speed_perturb: {}".format(speed_perturb))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_rank_data_list(self, data_index):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        if self.mode == "train":
            if self.shuffle:
                random.seed(self.epoch)
                random.shuffle(data_index)
            return data_index[self.rank::self.world_size]

        return data_index

    def get_worker_data_list(self, rank_data_index):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers

        return rank_data_index[self.worker_id::self.num_workers]

    def close_reader(self, reader_list):
        for reader in reader_list:
            reader.close()

    def __iter__(self):
        data_index = list(range(len(self.scp_lists)))
        rank_data_index = self.get_rank_data_list(data_index)
        worker_data_index = self.get_worker_data_list(rank_data_index)

        for index in worker_data_index:
            data = dict(scp=self.scp_lists[index])

            assert 'scp' in data
            scp = data['scp']
            data_file_list = scp.strip().split()
            data_name_list = self.data_names.split(",")
            data_type_list = self.data_types.split(",")

            for file in data_file_list:
                assert os.path.exists(file), "{} not exists".format(file)

            assert len(data_file_list) == len(data_name_list) == len(data_type_list), \
                "The item number of data, data_names, data_types must be the same "

            reader_list = []
            for data_file, data_type in zip(data_file_list, data_type_list):
                if data_type in ["kaldi_ark", "wav_ark"]:
                    ark_reader = ReadHelper('ark:{}'.format(data_file))
                    reader_list.append(ark_reader)
                elif data_type in ["text", "sound", "codec"]:
                    text_reader = open(data_file, "r")
                    reader_list.append(text_reader)
                elif data_type == "none":
                    continue
                else:
                    raise TypeError("Data type {} is not supported".format(data_type))

            for items in zip(*reader_list):
                sample_dict = {}
                for item, (data_name, data_type) in zip(items, zip(data_name_list, data_type_list)):
                    if data_name == "speech":
                        sample_dict["key"] = item[0]
                    if data_type in ["kaldi_ark", "wav_ark"]:
                        key, mat = item
                        sr = 16000
                        if isinstance(mat, tuple):
                            if isinstance(mat[0], int):
                                sr, mat = mat
                            else:
                                mat, sr = mat
                        if mat.dtype == np.int16:
                            mat = (mat / (2 ** 15)).astype(np.float32)
                        elif mat.dtype == np.int32:
                            mat = (mat / (2 ** 31)).astype(np.float32)
                        sample_dict["sampling_rate"] = sr
                        sample_dict[data_name] = mat
                    elif data_type == "sound":
                        key, path = item.strip().split()
                        waveform, sampling_rate = torchaudio.load(path)
                        if self.frontend_conf is not None:
                            if sampling_rate != self.frontend_conf["fs"]:
                                waveform = torchaudio.transforms.Resample(orig_freq=sampling_rate,
                                                                          new_freq=self.frontend_conf["fs"])(waveform)
                                sampling_rate = self.frontend_conf["fs"]
                        waveform = waveform.numpy()
                        mat = waveform[0]
                        if self.speed_perturb is not None:
                            speed = random.choice(self.speed_perturb)
                            if speed != 1.0:
                                mat, _ = torchaudio.sox_effects.apply_effects_tensor(
                                    torch.tensor(mat).view(1, -1), sampling_rate, [['speed', str(speed)], ['rate', str(sampling_rate)]])
                                mat = mat.view(-1).numpy()
                        sample_dict[data_name] = mat
                        sample_dict["sampling_rate"] = sampling_rate
                        if data_name == "speech":
                            sample_dict["key"] = key
                    elif data_type == "text_nospace":
                        text = item
                        segs = text.strip().split(maxsplit=1)
                        sample_dict[data_name] = [x for x in segs[1]]
                        if "key" not in sample_dict:
                            sample_dict["key"] = segs[0]
                    else:
                        text = item
                        segs = text.strip().split()
                        sample_dict[data_name] = segs[1:]
                        if "key" not in sample_dict:
                            sample_dict["key"] = segs[0]
                yield sample_dict

            self.close_reader(reader_list)


def len_fn_example(data):
    return 1


def len_fn_token(data):
    # assert "speech" in data
    if "sampling_rate" in data:
        return (data["speech"].shape[0] / data["sampling_rate"]) * 1000.
    elif "speech" in data:
        return data["speech"].shape[0]
    else:
        return data["text"].shape[0]


def len_fn_token_speech_text(data):
    return data["speech"].shape[0] + data["text"].shape[0]


def Dataset(data_list_file,
            vocab_dict,
            seg_dict,
            punc_dict,
            bpe_tokenizer,
            conf,
            frontend_conf,
            speed_perturb=None,
            mode="train",
            batch_mode="padding"):
    scp_lists = read_lists(data_list_file)
    shuffle = conf.get('shuffle', True)
    data_names = conf.get("data_names", "speech,text")
    data_types = conf.get("data_types", "kaldi_ark,text")
    speech_clip_conf = conf.get("speech_clip_conf", None)

    dataset = AudioDataset(
        scp_lists,
        data_names,
        data_types,
        frontend_conf=frontend_conf,
        shuffle=shuffle,
        speed_perturb=speed_perturb,
        mode=mode
    )

    filter_conf = conf.get('filter_conf', {})
    filter_fn = partial(filter, **filter_conf)
    dataset = FilterIterDataPipe(dataset, fn=filter_fn)

    if "text" in data_names:
        vocab = dict(
            vocab=vocab_dict,
            seg_dict=seg_dict,
            punc_dict=punc_dict,
            bpe_tokenizer=bpe_tokenizer,
        )
        tokenize_fn = partial(tokenize, **vocab)
        dataset = MapperIterDataPipe(dataset, fn=tokenize_fn)

    if speech_clip_conf is not None:
        clip_length_fn = partial(clip_speech_fix_length, **speech_clip_conf)
        dataset = MapperIterDataPipe(dataset, fn=clip_length_fn)

    if shuffle:
        buffer_conf = conf.get('shuffle_conf', {})
        buffer_size = buffer_conf['shuffle_size']
        sort_size = buffer_conf['sort_size']
    else:
        buffer_size = 0
        sort_size = 1

    batch_conf = conf.get('batch_conf', {})
    batch_size = batch_conf['batch_size']
    batch_type = batch_conf['batch_type']

    assert batch_type in ["example", "token", "token_speech_text"]
    if batch_type == 'example':
        len_fn = len_fn_example
    else:
        if batch_type == "token_speech_text":
            logging.info("using both speech and text for computing batch")
            len_fn = len_fn_token_speech_text
        else:
            len_fn = len_fn_token

    dataset = MaxTokenBucketizerIterDataPipe(dataset,
                                             batch_size=batch_size,
                                             len_fn=len_fn,
                                             buffer_size=buffer_size,
                                             sort_size=sort_size,
                                             batch_mode=batch_mode)

    int_pad_value = conf.get("int_pad_value", -1)
    float_pad_value = conf.get("float_pad_value", 0.0)
    padding_conf = {"int_pad_value": int_pad_value, "float_pad_value": float_pad_value}
    padding_fn = partial(padding, **padding_conf)
    dataset = MapperIterDataPipe(dataset, fn=padding_fn if batch_mode == "padding" else clipping)

    return dataset
