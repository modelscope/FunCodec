import numpy as np
import torch
from funcodec.datasets.collate_fn import crop_to_max_size


def clipping(data):
    assert isinstance(data, list)
    assert "key" in data[0]

    keys = [x["key"] for x in data]

    batch = {}
    data_names = data[0].keys()
    for data_name in data_names:
        if data_name == "key":
            continue
        else:
            if data[0][data_name].dtype.kind == "i":
                tensor_type = torch.int64
            else:
                tensor_type = torch.float32

            tensor_list = [torch.tensor(np.copy(d[data_name]), dtype=tensor_type) for d in data]
            tensor_lengths = torch.tensor([len(d[data_name]) for d in data], dtype=torch.int32)

            length_clip = min(tensor_lengths)
            tensor_clip = tensor_list[0].new_zeros(len(tensor_list), length_clip, tensor_list[0].shape[1])
            for i, (tensor, length) in enumerate(zip(tensor_list, tensor_lengths)):
                diff = length - length_clip
                assert diff >= 0
                if diff == 0:
                    tensor_clip[i] = tensor
                else:
                    tensor_clip[i] = crop_to_max_size(tensor, length_clip)

            batch[data_name] = tensor_clip
            batch[data_name + "_lengths"] = torch.tensor([tensor.shape[0] for tensor in tensor_clip], dtype=torch.long)

    return keys, batch


def clip_speech_fix_length(
        data,
        max_duration=3.2,  # in second
        frame_shift=0.01,  # in second
        sampling_rate=16000,
):
    assert isinstance(data, dict)
    assert "speech" in data
    if "sampling_rate" in data:
        sampling_rate = data["sampling_rate"]

    if max_duration > 0:
        speech = data["speech"]
        audio_length = len(speech)
        max_length = int(max_duration * sampling_rate)
        frame_shift = int(frame_shift * sampling_rate)
        if audio_length > max_length:
            max_start = audio_length - max_length
            start = np.random.randint(0, max_start, (1,))[0]
            speech = speech[start:start + max_length]
            if "ppg" in data:
                ppg = data["ppg"]
                st, dur = start / frame_shift, max_length / frame_shift
                data["ppg"] = ppg[int(st):int(st) + int(dur)]
            if "noisy_speech" in data:
                noisy = data["noisy_speech"]
                data["noisy_speech"] = noisy[start: start + max_length]
        else:
            speech = np.pad(speech, (0, max_length - audio_length))
            if "ppg" in data:
                pad_len = max_length / frame_shift - data["ppg"].shape[0]
                if len(data["ppg"].shape) == 2:
                    data["ppg"] = np.pad(data["ppg"], ((0, int(pad_len)), (0, 0)), mode="edge")
                else:
                    data["ppg"] = np.pad(data["ppg"], (0, int(pad_len)),
                                         mode="constant", constant_values=0)
            if "noisy_speech" in data:
                noisy = data["noisy_speech"]
                data["noisy_speech"] = np.pad(noisy, (0, max_length - audio_length))
        data["speech"] = speech

    return data
