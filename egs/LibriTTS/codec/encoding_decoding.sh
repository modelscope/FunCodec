#!/usr/bin/env bash

. ./path.sh || exit 1;

stage=1
wav_scp=
out_dir=
njob=1   # nj per GPU or all nj for CPU
gpu_devices="6,7"
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding
infer_cmd=utils/run.pl
model_dir=
sample_frequency=16000
file_sampling_rate=16000
bit_width=2000   # 8k: 16, 4k: 8, 2k: 4, 1k:2, 0.5k:1
need_indices=true
need_sub_quants=false
use_scale=false
batch_size=16
num_workers=4
data_format=
model_name=

. utils/parse_options.sh || exit 1;

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

# Downloading Stage
if [ ${stage} -eq 0 ]; then
  echo "stage 0: downloading model"
  git lfs install
  git clone https://www.modelscope.cn/damo/${model_name}.git
  mv ${model_name} exp/
fi

# Encoding Stage
if [ ${stage} -eq 1 ]; then
    echo "stage 1: encoding"

    _logdir="${out_dir}/logdir"
    if [ -d ${out_dir} ]; then
        echo "WARNING: ${out_dir} is already exists."
        exit 0
    fi
    mkdir -p "${_logdir}"
    key_file=${wav_scp}
    num_scp_file="$(<${key_file} wc -l)"
    _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
    split_scps=
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    if [ -z "$data_format" ]; then
      data_format="sound"
    fi
    utils/split_scp.pl "${key_file}" ${split_scps}
    ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/inference.JOB.log \
        python -m funasr.bin.codec_inference \
            --batch_size ${batch_size} \
            --num_workers ${num_workers} \
            --ngpu "${_ngpu}" \
            --gpuid_list ${gpuid_list} \
            --data_path_and_name_and_type "${wav_scp},speech,${data_format}" \
            --key_file "${_logdir}"/keys.JOB.scp \
            --config_file ${model_dir}/config.yaml \
            --model_file ${model_dir}/model.pth \
            --output_dir "${_logdir}"/output.JOB \
            --sampling_rate $sample_frequency \
            --file_sampling_rate $file_sampling_rate \
            --bit_width ${bit_width} \
            --need_indices true \
            --need_sub_quants false \
            --use_scale false  \
            --run_mod "encode"
fi

# Decoding Stage
if [ ${stage} -eq 2 ]; then
    echo "stage 2: Decoding"

    _logdir="${out_dir}/logdir"
    if [ -d ${out_dir} ]; then
        echo "WARNING: ${out_dir} is already exists."
        exit 0
    fi
    mkdir -p "${_logdir}"
    key_file=${wav_scp}
    num_scp_file="$(<${key_file} wc -l)"
    _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
    split_scps=
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    if [ -z "$data_format" ]; then
      data_format="codec_json"
    fi
    utils/split_scp.pl "${key_file}" ${split_scps}
    ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/inference.JOB.log \
        python -m funasr.bin.codec_inference \
            --batch_size ${batch_size} \
            --num_workers ${num_workers} \
            --ngpu "${_ngpu}" \
            --gpuid_list ${gpuid_list} \
            --data_path_and_name_and_type "${wav_scp},speech,${data_format}" \
            --key_file "${_logdir}"/keys.JOB.scp \
            --config_file ${model_dir}/config.yaml \
            --model_file ${model_dir}/model.pth \
            --output_dir "${_logdir}"/output.JOB \
            --sampling_rate $sample_frequency \
            --file_sampling_rate $file_sampling_rate \
            --bit_width ${bit_width} \
            --need_indices false \
            --need_sub_quants false \
            --use_scale false  \
            --run_mod "decode"
fi

# Decoding Stage
if [ ${stage} -eq 3 ]; then
    echo "stage 3: Decoding codec embeddings"

    _logdir="${out_dir}/logdir"
    if [ -d ${out_dir} ]; then
        echo "WARNING: ${out_dir} is already exists."
        exit 0
    fi
    mkdir -p "${_logdir}"
    key_file=${wav_scp}
    num_scp_file="$(<${key_file} wc -l)"
    _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
    split_scps=
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    if [ -z "$data_format" ]; then
      data_format="kaldi_ark"
    fi
    utils/split_scp.pl "${key_file}" ${split_scps}
    ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/inference.JOB.log \
        python -m funasr.bin.codec_inference \
            --batch_size ${batch_size} \
            --num_workers ${num_workers} \
            --ngpu "${_ngpu}" \
            --gpuid_list ${gpuid_list} \
            --data_path_and_name_and_type "${wav_scp},speech,${data_format}" \
            --key_file "${_logdir}"/keys.JOB.scp \
            --config_file ${model_dir}/config.yaml \
            --model_file ${model_dir}/model.pth \
            --output_dir "${_logdir}"/output.JOB \
            --sampling_rate $sample_frequency \
            --file_sampling_rate $file_sampling_rate \
            --bit_width ${bit_width} \
            --need_indices false \
            --need_sub_quants false \
            --use_scale false  \
            --run_mod "decode_emb"
fi
