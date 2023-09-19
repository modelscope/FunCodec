# FunCodec: A Fundamental, Reproducible and Integrable Open-source Toolkit for Neural Speech Codec

<strong>FunCodec</strong> 
This project is still working on progress.

## Installation

```shell
git clone https://github.com/alibaba/FunCodec.git && cd FunCodec
pip install --editable ./
```

## Available models
- Models on ModelScope

|                                                                          Model name                                                                          |  Corpus  |  Bitrate  |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------:|:---------:|
| [audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch](https://www.modelscope.cn/models/damo/audio_codec-encodec-en-libritts-16k-nq32ds320-pytorch/summary) | LibriTTS | 500~16000 |

- Models on Huggingface 

Models will be uploaded to Huggingface soon.


## Model Download
Please refer `egs/LibriTTS/codec/encoding_decoding.sh` to download models:
```shell
cd egs/LibriTTS/codec
bash encoding_decoding.sh --stage 0 --model_name audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch
# The pre-trained model will be downloaded to exp/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch
```

## Inference
Please refer `egs/LibriTTS/codec/encoding_decoding.sh` to perform encoding and decoding 
for a wave file list:
```shell
cd egs/LibriTTS/codec
bash encoding_decoding.sh --stage 1 --batch_size 16 --num_workers 4 --gpu_devices "0,1" \
  --model_dir exp/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch --bit_width 4000 \
  --wav_scp input_wav.scp  --out_dir outputs/codecs/
```

## Training
Please refer `egs/LibriTTS/codec/run.sh` to perform encoding and decoding 
for a wave file list.

## Acknowledge

1. We had a consistent design of [FunASR](https://github.com/alibaba/FunASR), including dataloader, model definition and so on.
2. We borrowed a lot of code from [Kaldi](http://kaldi-asr.org/) for data preparation.
3. We borrowed a lot of code from [ESPnet](https://github.com/espnet/espnet). FunCodec follows up the training and finetuning pipelines of ESPnet.

## License
This project is licensed under the [The MIT License](https://opensource.org/licenses/MIT). 
FunCodec also contains various third-party components and some code modified from other repos 
under other open source licenses.

## Citations

``` bibtex
@misc{du2023funcodec,
      title={FunCodec: A Fundamental, Reproducible and Integrable Open-source Toolkit for Neural Speech Codec},
      author={Zhihao Du, Shiliang Zhang, Kai Hu, Siqi Zheng},
      year={2023},
      eprint={2309.07405},
      archivePrefix={arXiv},
      primaryClass={cs.Sound}
}
```