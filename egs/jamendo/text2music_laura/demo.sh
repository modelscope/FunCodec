#!/usr/bin/env bash

. ./path.sh || exit 1;

stage=1
model_name="text2audio_codec_lm_nq2_uni_rel_pos_t5_enc_jamendo"
codec_model="audio_codec-freqcodec-universal-general-16k-nq32ds640-pytorch"
model_hub=modelscope
gpuid_list=0
text="genre: classical; instrument: piano, pianosolo; mood/theme: sadness"
prompt_text=""
prompt_audio="demo/03-1117703-0027.wav"
output_dir=
seed=0

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

if [ ! -d exp/t5-base ]; then
  mkdir -p exp
  git lfs install

  echo "downloading t5-base model from huggingface"
  git clone https://huggingface.co/t5-base

  mv t5-base exp/t5-base
fi

if [ ${stage} -eq 1 ]; then
  python -m funcodec.bin.text2audio_inference \
    --ngpu 1 --gpuid_list ${gpuid_list} \
    --config_file exp/${model_name}/config.yaml \
    --model_file exp/${model_name}/model.pth \
    --codec_config_file exp/${codec_model}/config.yaml \
    --codec_model_file exp/${codec_model}/model.pth \
    --text_emb_model exp/t5-base \
    --sampling 25 \
    --seed ${seed} \
    --log_level warning \
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
    --text_emb_model exp/t5-base \
    --sampling 25 \
    --seed ${seed} \
    --continual 25 \
    --exclude_prompt false \
    --log_level warning \
    --raw_inputs "${text}" \
    --raw_inputs "${prompt_text}" \
    --raw_inputs "${prompt_audio}" \
    --output_dir "${output_dir}"

  echo "Generated speeches are saved in ${output_dir}"
fi
