#!/usr/bin/env python
import re

import numpy as np

from funcodec.datasets.large_datasets.utils.hotword_utils import sample_hotword
import kaldiio

task_id_list = ["<ASR>", "<TTS>", "<MT>", "<S2TT>", "<SE>", "<AED>", "<ER>", "<LM>", "<DRVB>", "<ACP>"]


def forward_segment(text, seg_dict):
    word_list = []
    i = 0
    while i < len(text):
        longest_word = text[i]
        for j in range(i + 1, len(text) + 1):
            word = text[i:j]
            if word in seg_dict:
                if len(word) > len(longest_word):
                    longest_word = word
        word_list.append(longest_word)
        i += len(longest_word)
    return word_list


def seg_tokenize(txt, seg_dict):
    pattern = re.compile(r'^[\u4E00-\u9FA50-9]+$')
    out_txt = ""
    for word in txt:
        word = word.lower()
        if word in seg_dict:
            out_txt += seg_dict[word] + " "
        else:
            if pattern.match(word):
                for char in word:
                    if char in seg_dict:
                        out_txt += seg_dict[char] + " "
                    else:
                        out_txt += "<unk>" + " "
            else:
                out_txt += "<unk>" + " "
    return out_txt.strip().split()


def tokenize(data,
             vocab=None,
             seg_dict=None,
             punc_dict=None,
             bpe_tokenizer=None,
             hw_config=None):
    assert "text" in data
    assert isinstance(vocab, dict)
    text = data["text"]
    token = []
    vad = -2
    if bpe_tokenizer is not None:
        text = bpe_tokenizer.text2tokens(" ".join(text))
    if seg_dict is not None:
        assert isinstance(seg_dict, dict)
        text = seg_tokenize(text, seg_dict)

    length = len(text)
    if 'hw_tag' in data:
        hotword_indxs = sample_hotword(length, **hw_config)
        data['hotword_indxs'] = hotword_indxs
        del data['hw_tag']
    for i in range(length):
        x = text[i]
        if isinstance(x, str):
            if i == length - 1 and "punc" in data and x.startswith("vad:"):
                vad = x[4:]
                if len(vad) == 0:
                    vad = -1
                else:
                    vad = int(vad)
            elif x in vocab:
                token.append(vocab[x])
            else:
                token.append(vocab['<unk>'])
        else:
            token.append(x)

    if "punc" in data and punc_dict is not None:
        punc_token = []
        for punc in data["punc"]:
            if punc in punc_dict:
                punc_token.append(punc_dict[punc])
            else:
                punc_token.append(punc_dict["_"])
        data["punc"] = np.array(punc_token)

    data["text"] = np.array(token)
    if vad is not -2:
        data["vad_indexes"] = np.array([vad], dtype=np.int64)
    return data


# total dict: vocab + speech_codec + <task_id> + <sos> + <split> + <eos> + <unk>
# <eos> is also used for padding
def llm_tokenize(data,
                 vocab=None,
                 num_codes=1024,
                 num_groups=4,
                 num_tasks=16):
    input = data["input"]
    task_id = data["task_id"]
    text_vocab_size = len(vocab.keys())
    # num_tasks = len(task_id_list)
    num_codec = num_codes * num_groups
    # task_id_index = text_vocab_size + num_codec + task_id_list.index(task_id)
    sos_index = text_vocab_size + num_codec + num_tasks
    split_index = text_vocab_size + num_codec + num_tasks + 1
    eos_index = text_vocab_size + num_codec + num_tasks + 2
    padding_index = eos_index
    unk_index = text_vocab_size + num_codec + num_tasks + 3
    token = [sos_index]
    # add <sos>
    if task_id in ["<TTS>", "<MT>", "<LM>", "<ER>"]:
        parts = input.split()
        for p in parts:
            if p in vocab:
                token.append(vocab[p])
            else:
                token.append(unk_index)
    else:
        shift_group = np.arange(num_groups) * num_codes
        input = np.array(input) + shift_group[None, :]
        codec_token = (input.reshape(-1) + text_vocab_size).tolist()  # (T, num_groups)
        token = token + codec_token
    token.append(split_index)
    new_data = {}
    new_data["key"] = data["key"]
    new_data["text"] = np.array(token)
    new_data["padding_index"] = padding_index

    return new_data


# <sos> + input + <task_id> + output + <eos>
def llm_sft_tokenize(data,
                     vocab=None,
                     num_codes=1024,
                     num_groups=4,
                     num_tasks=16):
    input = data["input"]
    output = data["output"]
    task_id = data["task_id"]
    text_vocab_size = len(vocab.keys())
    # num_tasks = len(task_id_list)
    num_codec = num_codes * num_groups
    task_id_index = text_vocab_size + num_codec + task_id_list.index(task_id)
    if "<s>" in vocab.keys():
        sos_index = vocab["<s>"]
    else:
        sos_index = text_vocab_size + num_codec + num_tasks
    split_index = text_vocab_size + num_codec + num_tasks + 1
    if "</s>" in vocab.keys():
        eos_index = vocab["</s>"]
    else:
        eos_index = text_vocab_size + num_codec + num_tasks + 2
    padding_index = eos_index
    if "<unk>" in vocab.keys():
        unk_index = vocab["<unk>"]
    else:
        unk_index = text_vocab_size + num_codec + num_tasks + 3

    new_data = {}
    # add <sos>
    token = [sos_index]
    # add input
    if task_id in ["<TTS>", "<MT>", "<LM>", "<ER>"]:
        parts = input.split()
        for p in parts:
            if p in vocab:
                token.append(vocab[p])
            else:
                token.append(unk_index)
    else:
        shift_group = np.arange(num_groups) * num_codes
        input = np.array(input) + shift_group[None, :]
        codec_token = (input.reshape(-1) + text_vocab_size).tolist()  # (T, num_groups)
        token = token + codec_token
    # add task_id
    token.append(task_id_index)
    # save input length for calc output acc
    new_data["input_length"] = len(token)
    # add output
    if task_id in ["<ASR>", "<MT>", "<S2TT>", "<AED>", "<ER>", "<LM>", "<ACP>"]:
        output_parts = output.split()
        for p in output_parts:
            if p in vocab:
                token.append(vocab[p])
            else:
                token.append(unk_index)
    else:
        shift_group = np.arange(num_groups) * num_codes
        output = np.array(output) + shift_group[None, :]
        codec_token = (output.reshape(-1) + text_vocab_size).tolist()  # (T, num_groups)
        token = token + codec_token
    # add <eos>
    token.append(eos_index)

    new_data["key"] = data["key"]
    new_data["text"] = np.array(token)
    new_data["padding_index"] = padding_index

    return new_data

# <sos> + input + <task_id> + output + <eos>
def llm_sft_tokenize_v2(data,
                     vocab=None,
                     num_codes=1024,
                     num_groups=4,
                     num_tasks=16):
    # unk_index = 0
    # sos_index = 1
    # eos_index = 2
    # padding_index = 3
    # split_index = 4
    unk_index = vocab["<unk>"]
    sos_index = vocab["<s>"]
    eos_index = vocab["</s>"]
    padding_index = vocab["<pad>"]
    split_index = vocab["<split>"]
    input = data["input"]
    output = data["output"]
    task_id = data["taskid"] if "taskid" in data else data["task_id"]
    task_id_index = vocab[task_id]
    text_vocab_size = 20003
    # num_tasks = len(task_id_list)
    num_codec = num_codes * num_groups
    # task_id_index = 5 + text_vocab_size + num_codec + task_id_list.index(task_id)

    # sos_index = text_vocab_size + num_codec + num_tasks
    # split_index = text_vocab_size + num_codec + num_tasks + 1
    # eos_index = text_vocab_size + num_codec + num_tasks + 2
    # padding_index = eos_index
    # unk_index = text_vocab_size + num_codec + num_tasks + 3
    token = [sos_index]
    if task_id in ["<TTS>", "<MT>", "<LM>", "<ER>"]:
        if isinstance(input, list):
            parts = input
        else:
            parts = input.split()
        for p in parts:
            p = str(p)
            if p in vocab:
                token.append(vocab[p])
                # print("<TTS>", "<MT>", "<LM>: ", token_id_new)
            else:
                token.append(unk_index)
    else:
        shift_group = np.arange(num_groups) * num_codes
        input = np.array(input) + shift_group[None, :]

        codec_token = (input.reshape(-1) + text_vocab_size).tolist()  # (T, num_groups)
        token = token + codec_token
        # print("codec: ", token)

    token.append(task_id_index)
    new_data = {}
    new_data["input_length"] = len(token)

    if task_id in ["<ASR>", "<MT>", "<S2TT>", "<AED>", "<ER>", "<LM>", "<ACP>"]:
        if isinstance(output, list):
            parts = output
        else:
            parts = output.split()
        for p in parts:
            p = str(p)
            if p in vocab:
                token.append(vocab[p])
            else:
                token.append(unk_index)
    else:
        shift_group = np.arange(num_groups) * num_codes
        input = np.array(output) + shift_group[None, :]

        codec_token = (input.reshape(-1) + text_vocab_size).tolist()  # (T, num_groups)
        token = token + codec_token
        # print("codec: ", token)

    token.append(eos_index)

    new_data["key"] = data["key"]
    new_data["text"] = np.array(token)
    new_data["padding_index"] = padding_index

    return new_data

# Similar with v2, but return input and output respectively
# input: no sos, no task_id
# output: with task_id, no eos
def llm_sft_tokenize_v3(data,
                     vocab=None,
                     num_codes=1024,
                     num_groups=4,
                     num_tasks=16):
    # unk_index = 0
    # sos_index = 1
    # eos_index = 2
    # padding_index = 3
    # split_index = 4
    unk_index = vocab["<unk>"]
    sos_index = vocab["<s>"]
    eos_index = vocab["</s>"]
    padding_index = vocab["<pad>"]
    split_index = vocab["<split>"]
    input = data["input"]
    output = data["output"]
    task_id = data["taskid"] if "taskid" in data else data["task_id"]
    task_id_index = vocab[task_id]
    text_vocab_size = 20003
    # num_tasks = len(task_id_list)
    num_codec = num_codes * num_groups
    # task_id_index = 5 + text_vocab_size + num_codec + task_id_list.index(task_id)

    # sos_index = text_vocab_size + num_codec + num_tasks
    # split_index = text_vocab_size + num_codec + num_tasks + 1
    # eos_index = text_vocab_size + num_codec + num_tasks + 2
    # padding_index = eos_index
    # unk_index = text_vocab_size + num_codec + num_tasks + 3
    # token = [sos_index]
    input_token = []
    if task_id in ["<TTS>", "<MT>", "<LM>", "<ER>"]:
        if isinstance(input, list):
            parts = input
        else:
            parts = input.split()
        for p in parts:
            p = str(p)
            if p in vocab:
                input_token.append(vocab[p])
                # print("<TTS>", "<MT>", "<LM>: ", token_id_new)
            else:
                input_token.append(unk_index)
    else:
        shift_group = np.arange(num_groups) * num_codes
        input = np.array(input) + shift_group[None, :]

        input_token = (input.reshape(-1) + text_vocab_size).tolist()  # (T, num_groups)
        # input_token = input_token + codec_token
        # print("codec: ", token)

    output_token = [task_id_index]
    # token.append(task_id_index)
    new_data = {}
    # new_data["input_length"] = len(token)

    if task_id in ["<ASR>", "<MT>", "<S2TT>", "<AED>", "<ER>", "<LM>", "<ACP>"]:
        if isinstance(output, list):
            parts = output
        else:
            parts = output.split()
        for p in parts:
            p = str(p)
            if p in vocab:
                output_token.append(vocab[p])
            else:
                output_token.append(unk_index)
    else:
        shift_group = np.arange(num_groups) * num_codes
        input = np.array(output) + shift_group[None, :]

        codec_token = (input.reshape(-1) + text_vocab_size).tolist()  # (T, num_groups)
        output_token = output_token + codec_token
        # print("codec: ", token)

    # token.append(eos_index)

    new_data["key"] = data["key"]
    # new_data["text"] = np.array(token)
    new_data["padding_index"] = padding_index

    # speech means input, text means output
    # These two can be both codec and text
    new_data["speech"] = np.array(input_token)
    new_data["text"] = np.array(output_token)

    return new_data

# Similar with v3, the difference is as follows:
# 1. when input is speech, use LFR6 fbank instead of codec
# 2. when output is speech, only use the first group of codec
# 3. is_fbank_flag and is_not_fbank_flag are also returned now
# 4. output includes <task_id> and <eos>
def llm_enc_sft_tokenize(data,
                     vocab=None,
                     num_codes=1024,
                     num_groups=4,
                     num_tasks=16):
    # unk_index = 0
    # sos_index = 1
    # eos_index = 2
    # padding_index = 3
    # split_index = 4
    unk_index = vocab["<unk>"]
    sos_index = vocab["<s>"]
    eos_index = vocab["</s>"]
    padding_index = vocab["<pad>"]
    split_index = vocab["<split>"]
    input = data["input"]
    output = data["output"]
    task_id = data["taskid"] if "taskid" in data else data["task_id"]
    task_id_index = vocab[task_id]
    text_vocab_size = 20003
    # num_tasks = len(task_id_list)
    num_codec = num_codes * num_groups
    # task_id_index = 5 + text_vocab_size + num_codec + task_id_list.index(task_id)

    # sos_index = text_vocab_size + num_codec + num_tasks
    # split_index = text_vocab_size + num_codec + num_tasks + 1
    # eos_index = text_vocab_size + num_codec + num_tasks + 2
    # padding_index = eos_index
    # unk_index = text_vocab_size + num_codec + num_tasks + 3
    # token = [sos_index]
    input_token = []
    if task_id in ["<TTS>", "<MT>", "<LM>", "<ER>"]:
        # if isinstance(input, list):
        #     parts = input
        # else:
        #     parts = input.split()
        # for p in parts:
        #     p = str(p)
        #     if p in vocab:
        #         input_token.append(vocab[p])
        #     else:
        #         input_token.append(unk_index)
        parts = input.tolist()
        for p in parts:
            p = str(p)
            if p in vocab:
                input_token.append(vocab[p])
            else:
                input_token.append(unk_index)
    else:
        # input_token = kaldiio.load_mat(input)
        input_token = input

    output_token = [task_id_index]
    new_data = {}
    # new_data["input_length"] = len(token)

    if task_id in ["<ASR>", "<MT>", "<S2TT>", "<AED>", "<ER>", "<LM>", "<ACP>"]:
        if isinstance(output, list):
            parts = output
        else:
            parts = output.split()
        for p in parts:
            p = str(p)
            if p in vocab:
                output_token.append(vocab[p])
            else:
                output_token.append(unk_index)
    else:
        # shift_group = np.arange(num_groups) * num_codes
        # input = np.array(output) + shift_group[None, :]

        # codec_token = (input.reshape(-1) + text_vocab_size).tolist()  # (T, num_groups)
        # output_token = output_token + codec_token
        # print("codec: ", token)

        # only use the first group of codec now
        codec_token = (np.array(output)[:, 0] + text_vocab_size).tolist()  # (T, num_groups) -> (T, )
        output_token = output_token + codec_token

    output_token.append(eos_index)

    new_data["key"] = data["key"]
    # new_data["text"] = np.array(token)
    new_data["padding_index"] = padding_index
    new_data["is_fbank_flag"] = True if task_id not in ["<TTS>", "<MT>", "<LM>", "<ER>"] else False

    # speech means input, text means output
    # These two can be both codec and text
    new_data["speech"] = np.array(input_token)
    new_data["text"] = np.array(output_token)

    return new_data