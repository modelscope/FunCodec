#!/usr/bin/env bash

. ./path.sh || exit 1;

# global configs
stage=1

# pre-trained related
model_name=
model_hub=modelscope
codec_model="audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch"

# training related
train_config="conf/"
tag=""

# inference related
text_scp=
out_dir=
njob=1   # nj per GPU or all nj for CPU
gpu_devices="6,7"
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding
num_workers=4
infer_cmd=utils/run.pl

. utils/parse_options.sh || exit 1;

if [ ! -z ${model_name} ]; then
  model_dir="exp/${model_name}"
else
  model_dir="exp/$(basename "${train_config}" .yaml)${tag}"
fi

# you can set gpu num for decoding here
gpuid_list=$gpu_devices  # set gpus for decoding, the same as training stage by default
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')

if ${gpu_inference}; then
    inference_nj=$[${ngpu}*${njob}]
    _ngpu=1
else
    inference_nj=$njob
    _ngpu=0
fi

# stage 0: download pre-trained model
if [ ${stage} -eq 0 ]; then
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
fi

# stage 1: download and preprocess dataset
if [ ${stage} -eq 1 ]; then
  echo "stage 1: prepare dataset"
fi

# stage 2: training model
if [ ${stage} -eq 2 ]; then
  echo "stage 2: training"
fi

# stage 3: inference
if [ ${stage} -eq 3 ]; then
    echo "stage 3: inference"

    _logdir="${out_dir}/logdir"
    if [ -d ${out_dir} ]; then
        echo "ERROR: ${out_dir} is already exists."
        exit 0
    fi
    mkdir -p "${_logdir}"
    key_file=${text_scp}
    num_scp_file="$(<${key_file} wc -l)"
    _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
    split_scps=
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/keys.${n}.scp"
    done

    utils/split_scp.pl "${key_file}" ${split_scps}
    ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/inference.JOB.log \
        python -m funcodec.bin.text2audio_inference \
            --batch_size 1 \
            --num_workers ${num_workers} \
            --ngpu "${_ngpu}" \
            --gpuid_list ${gpuid_list} \
            --data_path_and_name_and_type "${text_scp},text,text" \
            --key_file "${_logdir}"/keys.JOB.scp \
            --config_file ${model_dir}/config.yaml \
            --model_file ${model_dir}/model.pth \
            --output_dir "${_logdir}"/output.JOB \
            --codec_config_file exp/${codec_model}/config.yaml \
            --codec_model_file exp/${codec_model}/model.pth \
            --sampling 25 \
            --tokenize_to_phone true

    echo "Generated speeches are saved to ${_logdir}/output.*/*.wav"

fi
