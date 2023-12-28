#!/usr/bin/env bash

. ./path.sh || exit 1;

stage=1
model_name="speech_synthesizer-laura-en-libritts-16k-codec_nq2-pytorch"
codec_model="audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch"
model_hub=modelscope
gpuid_list=0
text="nothing was to be done but to put about, and return in disappointment towards the north."
prompt_text="one of these is context"                # "nothing was to be done"
prompt_audio="demo/8230_279154_000013_000003.wav"    # demo/5105_28241_000027_000002.wav
output_dir=

. utils/parse_options.sh || exit 1;

if [ ! -d exp/${model_name} ]; then
  mkdir -p exp
  git lfs install
  if [ "${model_hub}" = "modelscope" ]; then
    echo "downloading generation model from modelscope"
    git clone https://www.modelscope.cn/damo/${model_name}.git
  fi

  if [ "${model_hub}" = "huggingface" ]; then
    echo "downloading generation model from huggingface"
    git clone https://huggingface.co/alibaba-damo/${model_name}
  fi

  mv ${model_name} exp/${model_name}
fi

if [ ! -d exp/${codec_model} ]; then
  mkdir -p exp
  git lfs install

  if [ "${model_hub}" = "modelscope" ]; then
    echo "downloading codec model from modelscope"
    git clone https://www.modelscope.cn/damo/${codec_model}.git
  fi

  if [ "${model_hub}" = "huggingface" ]; then
    echo "downloading codec model from huggingface"
    git clone https://huggingface.co/alibaba-damo/${codec_model}
  fi

  mv ${codec_model} exp/${codec_model}
fi

if [ ${stage} -eq 1 ]; then
  python -m funcodec.bin.text2audio_inference \
    --ngpu 1 --gpuid_list ${gpuid_list} \
    --config_file exp/${model_name}/config.yaml \
    --model_file exp/${model_name}/model.pth \
    --codec_config_file exp/${codec_model}/config.yaml \
    --codec_model_file exp/${codec_model}/model.pth \
    --sampling 25 \
    --log_level warning \
    --tokenize_to_phone true \
    --raw_inputs "${text}" \
    --output_dir "${output_dir}"

  echo "Generated speeches are saved in ${output_dir}"
fi

if [ ${stage} -eq 2 ]; then
  python -m funcodec.bin.text2audio_inference \
    --ngpu 1 --gpuid_list ${gpuid_list} \
    --config_file exp/${model_name}/config.yaml \
    --model_file exp/${model_name}/model.pth \
    --codec_config_file exp/${codec_model}/config.yaml \
    --codec_model_file exp/${codec_model}/model.pth \
    --sampling 25 \
    --continual 2500 \
    --log_level warning \
    --tokenize_to_phone true \
    --raw_inputs "${text}" \
    --raw_inputs "${prompt_text}" \
    --raw_inputs "${prompt_audio}" \
    --output_dir "${output_dir}"

  echo "Generated speeches are saved in ${output_dir}"
fi
