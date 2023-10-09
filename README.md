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

| Model name                                                                                                                                                                               |   Corpus    |  Bitrate  | Parameters | Flops  |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------:|:---------:|:----------:|:------:|
| [audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch](https://www.modelscope.cn/models/damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/summary)                         |   General   | 250~8000  |  57.83 M   | 7.73G  |
| [audio_codec-encodec-zh_en-general-16k-nq32ds320-pytorch](https://www.modelscope.cn/models/damo/audio_codec-encodec-zh_en-general-16k-nq32ds320-pytorch/summary)                         |   General   | 500~16000 |  14.85 M   | 3.72 G |
| [audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch](https://www.modelscope.cn/models/damo/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch/summary)                             |  LibriTTS   | 250~8000  |  57.83 M   | 7.73G  |
| [audio_codec-encodec-en-libritts-16k-nq32ds320-pytorch](https://www.modelscope.cn/models/damo/audio_codec-encodec-en-libritts-16k-nq32ds320-pytorch/summary)                             |  LibriTTS   | 500~16000 |  14.85 M   | 3.72 G |
| [audio_codec-freqcodec_magphase-en-libritts-16k-gr8nq32ds320-pytorch](https://www.modelscope.cn/models/damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr8nq32ds320-pytorch/summary) |  LibriTTS   | 500~16000 |   4.50 M   | 2.18 G | 
| [audio_codec-freqcodec_magphase-en-libritts-16k-gr1nq32ds320-pytorch](https://www.modelscope.cn/models/damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr1nq32ds320-pytorch/summary) |  LibriTTS   | 500~16000 |   0.52 M   | 0.34 G |


- Models on Huggingface 

Models will be uploaded to Huggingface soon.


## Model Download
Please refer `egs/LibriTTS/codec/encoding_decoding.sh` to download models:
```shell
cd egs/LibriTTS/codec
model_name=audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch
bash encoding_decoding.sh --stage 0 --model_name ${model_name}
# The pre-trained model will be downloaded to exp/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch
```

## Inference
Please refer `egs/LibriTTS/codec/encoding_decoding.sh` to perform encoding and decoding.
Extract codes with an input file `input_wav.scp`, 
and the codes will be saved to `output_dir/codecs.txt` in a format of jsonl.
```shell
cd egs/LibriTTS/codec
bash encoding_decoding.sh --stage 1 --batch_size 16 --num_workers 4 --gpu_devices "0,1" \
  --model_dir exp/${model_name} --bit_width 4000 \
  --wav_scp input_wav.scp  --out_dir outputs/codecs/
# input_wav.scp has the following format：
# uttid1 path/to/file1.wav
# uttid2 path/to/file2.wav
# ...
```

Decode codes with an input file `codecs.txt`, 
and the reconstructed waveform will be saved to `output_dir/logdir/output.*/*.wav`.
```shell
bash encoding_decoding.sh --stage 2 --batch_size 16 --num_workers 4 --gpu_devices "0,1" \
  --model_dir exp/${model_name} --bit_width 8000 --file_sampling_rate 16000 \
  --wav_scp codecs.txt --out_dir outputs/recon_wavs 
# codecs.scp is the output of above encoding stage, which has the following format：
# uttid1 [[[1, 2, 3, ...],[2, 3, 4, ...], ...]]
# uttid2 [[[9, 7, 5, ...],[3, 1, 2, ...], ...]]
```

## Training
Please refer `egs/LibriTTS/codec/run.sh` to perform training.

## Acknowledge

1. We had a consistent design of [FunASR](https://github.com/alibaba/FunASR), including dataloader, model definition and so on.
2. We borrowed a lot of code from [Kaldi](http://kaldi-asr.org/) for data preparation.
3. We borrowed a lot of code from [ESPnet](https://github.com/espnet/espnet). FunCodec follows up the training and finetuning pipelines of ESPnet.

## License
This project is licensed under [The MIT License](https://opensource.org/licenses/MIT). 
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
