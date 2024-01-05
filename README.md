# FunCodec: A Fundamental, Reproducible and Integrable Open-source Toolkit for Neural Speech Codec

This project is still working on progress. To make FunCodec better, please let me know your concerns and feel free to comment them in the `Issues` part.

## News
- 2023.12.22 🎉🎉: We release the training and inference recipes for LauraTTS as well as pre-trained models. 
[LauraTTS](https://arxiv.org/abs/2310.04673) is a powerful codec-based zero-shot text-to-speech synthesizer, 
which outperforms VALL-E in terms of semantic consistency and speaker similarity.
Please refer `egs/LibriTTS/text2speech_laura/README.md` for more details.

## Installation

```shell
git clone https://github.com/alibaba/FunCodec.git && cd FunCodec
pip install --editable ./
```

## Available models
🤗 links to the Huggingface model hub, while ⭐ refers the Modelscope.

| Model name                                                          |                                                                                                              Model hub                                                                                                               |  Corpus  |  Bitrate  | Parameters | Flops  |
|:--------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------:|:---------:|:----------:|:------:|
| audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch             |             [🤗](https://huggingface.co/alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch) [⭐](https://www.modelscope.cn/models/damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/summary)             | General  | 250~8000  |  57.83 M   | 7.73G  |
| audio_codec-encodec-zh_en-general-16k-nq32ds320-pytorch             |             [🤗](https://huggingface.co/alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds320-pytorch) [⭐](https://www.modelscope.cn/models/damo/audio_codec-encodec-zh_en-general-16k-nq32ds320-pytorch/summary)             | General  | 500~16000 |  14.85 M   | 3.72 G |
| audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch               |               [🤗](https://huggingface.co/alibaba-damo/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch) [⭐](https://www.modelscope.cn/models/damo/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch/summary)               | LibriTTS | 250~8000  |  57.83 M   | 7.73G  |
| audio_codec-encodec-en-libritts-16k-nq32ds320-pytorch               |               [🤗](https://huggingface.co/alibaba-damo/audio_codec-encodec-en-libritts-16k-nq32ds320-pytorch) [⭐](https://www.modelscope.cn/models/damo/audio_codec-encodec-en-libritts-16k-nq32ds320-pytorch/summary)               | LibriTTS | 500~16000 |  14.85 M   | 3.72 G |
| audio_codec-freqcodec_magphase-en-libritts-16k-gr8nq32ds320-pytorch | [🤗](https://huggingface.co/alibaba-damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr8nq32ds320-pytorch) [⭐](https://www.modelscope.cn/models/damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr8nq32ds320-pytorch/summary) | LibriTTS | 500~16000 |   4.50 M   | 2.18 G | 
| audio_codec-freqcodec_magphase-en-libritts-16k-gr1nq32ds320-pytorch | [🤗](https://huggingface.co/alibaba-damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr1nq32ds320-pytorch) [⭐](https://www.modelscope.cn/models/damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr1nq32ds320-pytorch/summary) | LibriTTS | 500~16000 |   0.52 M   | 0.34 G |

## Model Download
### Download models from ModelScope
Please refer `egs/LibriTTS/codec/encoding_decoding.sh` to download pretrained models:
```shell
cd egs/LibriTTS/codec
model_name=audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch
bash encoding_decoding.sh --stage 0 --model_name ${model_name} --model_hub modelscope
# The pre-trained model will be downloaded to exp/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch
```

### Download models from Huggingface
Please refer `egs/LibriTTS/codec/encoding_decoding.sh` to download pretrained models:
```shell
cd egs/LibriTTS/codec
model_name=audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch
bash encoding_decoding.sh --stage 0 --model_name ${model_name} --model_hub huggingface
# The pre-trained model will be downloaded to exp/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch
```

## Inference
### Batch inference
Please refer `egs/LibriTTS/codec/encoding_decoding.sh` to perform encoding and decoding.
Extract codes with an input file `input_wav.scp`, 
and the codes will be saved to `output_dir/codecs.txt` in a format of jsonl.
```shell
cd egs/LibriTTS/codec
bash encoding_decoding.sh --stage 1 --batch_size 16 --num_workers 4 --gpu_devices "0,1" \
  --model_dir exp/${model_name} --bit_width 16000 \
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
  --model_dir exp/${model_name} --bit_width 16000 --file_sampling_rate 16000 \
  --wav_scp codecs.txt --out_dir outputs/recon_wavs 
# codecs.scp is the output of above encoding stage, which has the following format：
# uttid1 [[[1, 2, 3, ...],[2, 3, 4, ...], ...]]
# uttid2 [[[9, 7, 5, ...],[3, 1, 2, ...], ...]]
```

<!---
### Demo inference
--->

## Training
### Training on open-source corpora
For commonly-used open-source corpora, you can train a model using the recipe in `egs` directory.
For example, to train a model on the `LibriTTS` corpus, you can use `egs/LibriTTS/codec/run.sh`:
```shell
# entry the LibriTTS recipe directory
cd egs/LibriTTS/codec
# run data downloading, preparation and training stages with 2 GPUs (device 0 and 1)
bash run.sh --stage 0 --stop_stage 3 --gpu_devices 0,1 --gpu_num 2
```
We recommend run the script stage by stage to have an overview of FunCodec.

### Training on customized data
For uncovered corpora or customized dataset, you can prepare the data by yourself.
In general, FunCodec employs the kaldi-like `wav.scp` file to organize the data files.
`wav.scp` has the following format:
```shell
# for waveform files
uttid1 /path/to/uttid1.wav
uttid2 /path/to/uttid2.wav
# for kaldi-ark files
uttid3 /path/to/ark1.wav:10
uttid4 /path/to/ark1.wav:200
uttid5 /path/to/ark2.wav:10
```
As shown in the above example, FunCodec supports the combination of waveforms or kaldi-ark files 
in one `wav.scp` file for both training and inference.
Here is a demo script to train a model on your customized dataset named `foo`:
```shell
cd egs/LibriTTS/codec
# 0. make the directory for train, dev and test sets
mkdir -p dump/foo/train dump/foo/dev dump/foo/test

# 1a. if you already have the wav.scp file, just place them under the corresponding directories
mv train.scp dump/foo/train/; mv dev.scp dump/foo/dev/; mv test.scp dump/foo/test/;
# 1b. if you don't have the wav.scp file, you can prepare it as follows
find path/to/train_set/ -iname "*.wav" | awk -F '/' '{print $(NF),$0}' | sort > dump/foo/train/wav.scp
find path/to/dev_set/   -iname "*.wav" | awk -F '/' '{print $(NF),$0}' | sort > dump/foo/dev/wav.scp
find path/to/test_set/  -iname "*.wav" | awk -F '/' '{print $(NF),$0}' | sort > dump/foo/test/wav.scp

# 2. collate shape files
mkdir exp/foo_states/train exp/foo_states/dev
torchrun --nproc_per_node=4 --master_port=1234 scripts/gen_wav_length.py --wav_scp dump/foo/train/wav.scp --out_dir exp/foo_states/train/wav_length
cat exp/foo_states/train/wav_length/wav_length.*.txt | shuf > exp/foo_states/train/speech_shape
torchrun --nproc_per_node=4 --master_port=1234 scripts/gen_wav_length.py --wav_scp dump/foo/dev/wav.scp --out_dir exp/foo_states/dev/wav_length
cat exp/foo_states/dev/wav_length/wav_length.*.txt | shuf > exp/foo_states/dev/speech_shape

# 3. train the model with 2 GPUs (device 4 and 5) on the customized dataset (foo)
bash run.sh --gpu_devices 4,5 --gpu_num 2 --dumpdir dump/foo --state_dir foo_states
```

## Acknowledge

1. We had a consistent design of [FunASR](https://github.com/alibaba/FunASR), including dataloader, model definition and so on.
2. We borrowed a lot of code from [Kaldi](http://kaldi-asr.org/) for data preparation.
3. We borrowed a lot of code from [ESPnet](https://github.com/espnet/espnet). FunCodec follows up the training and finetuning pipelines of ESPnet.
4. We borrowed the design of model architecture from [Enocdec](https://github.com/facebookresearch/encodec) and [Enocdec_Trainner](https://github.com/Mikxox/EnCodec_Trainer).

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
