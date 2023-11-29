#!/usr/bin/env bash

. ./path.sh || exit 1;

# global configs
stage=1
stop_stage=1

# data related
corpus_dir=corpus/LibriTTS
dumpdir=dump/libritts
state_dir=exp/libritts_states

# pre-trained related
model_name=
model_hub=modelscope
codec_model="audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch"

# training related
train_config="conf/"
tag=""
feats_dir="."
exp_dir="."

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
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
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

# Data downloading
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: data downloading"

  if [ ! -d ${corpus_dir} ]; then
    mkdir -p ${corpus_dir}
  fi

  echo "download training set to ${corpus_dir}"
  wget --no-check-certificate https://www.openslr.org/resources/60/train-clean-100.tar.gz -P ${corpus_dir}/
  wget --no-check-certificate https://www.openslr.org/resources/60/train-clean-360.tar.gz -P ${corpus_dir}/
  wget --no-check-certificate https://www.openslr.org/resources/60/train-other-500.tar.gz -P ${corpus_dir}/

  echo "download dev set to ${corpus_dir}"
  wget --no-check-certificate https://www.openslr.org/resources/60/dev-clean.tar.gz -P ${corpus_dir}/
  wget --no-check-certificate https://www.openslr.org/resources/60/dev-other.tar.gz -P ${corpus_dir}/

  echo "download test set to ${corpus_dir}"
  wget --no-check-certificate https://www.openslr.org/resources/60/test-clean.tar.gz -P ${corpus_dir}/
  wget --no-check-certificate https://www.openslr.org/resources/60/test-other.tar.gz -P ${corpus_dir}/

  cd ${corpus_dir}/
  tar zxf train-clean-100.tar.gz train-clean-360.tar.gz train-other-500.tar.gz
  tar zxf dev-clean.tar.gz dev-other.tar.gz
  tar zxf test-clean.tar.gz test-other.tar.gz

  # remove the duplicated LibriTTS directory
  mv ${corpus_dir}/LibriTTS/* ${corpus_dir}/
  rm -rf ${corpus_dir}/LibriTTS
fi

# Data collecting
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: collecting data sets."
  mkdir -p ${dumpdir}/train_24k ${dumpdir}/dev_24k

  for name in train-clean-100 train-clean-360 train-other-500; do
    echo "collecting ${name} in to ${dumpdir}/train_24k"
    find ${corpus_dir}/${name}/ -iname "*.wav" | awk -F '[/.]' '{print $(NF-1), $0}' | sort >> ${dumpdir}/train_24k/wav.scp
    find ${corpus_dir}/${name}/ -iname "*.normalized.txt" | awk -F '[/.]' '{print $(NF-2),$0}' | sort >> ${dumpdir}/train_24k/normalized_txt.flist
  done

  for name in dev-clean dev-other; do
    echo "collecting ${name} in to ${dumpdir}/dev_24k"
    find ${corpus_dir}/${name}/ -iname "*.wav" | awk -F '[/.]' '{print $(NF-1), $0}' | sort >> ${dumpdir}/dev_24k/wav.scp
    find ${corpus_dir}/${name}/ -iname "*.normalized.txt" | awk -F '[/.]' '{print $(NF-2),$0}' | sort >> ${dumpdir}/dev_24k/normalized_txt.flist
  done

  for name in test-clean test-other; do
    mkdir -p ${dumpdir}/${name}_24k
    echo "collecting ${name} in to ${dumpdir}/${name}_24k"
    find ${corpus_dir}/${name}/ -iname "*.wav" | awk -F '[/.]' '{print $(NF-1), $0}' | sort > ${dumpdir}/${name}_24k/wav.scp
    find ${corpus_dir}/${name}/ -iname "*.normalized.txt" | awk -F '[/.]' '{print $(NF-2),$0}' | sort >> ${dumpdir}/${name}_24k/normalized_txt.flist
  done
fi

# Dump data to ark and convert it to the sampling rate of 16000
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: Dump data to ark."
  for name in train dev; do
    echo "Dump ${name} set to ark files ${dumpdir}/${name}/arks/wav.*.ark"
    torchrun --nproc_per_node=32 --master_port=1234 scripts/dump_to_wav_ark.py \
      --wav_scp ${dumpdir}/${name}_24k/wav.scp \
      --out_dir ${dumpdir}/${name}/arks \
      --sample_rate 16000

    mkdir -p ${dumpdir}/${name} exp/${state_dir}/${name}
    cat ${dumpdir}/${name}/arks/wav.*.scp | sort > ${dumpdir}/${name}/wav.scp
    cat ${dumpdir}/${name}/arks/length.*.txt | shuf | awk '{print $1,int($2/640)}' > exp/${state_dir}/${name}/codec_shape

    echo "Collect and tokenize text files of ${name} into one phoneme file"
    python scripts/collect_text_flist_to_phone_scp.py \
      ${dumpdir}/${name}_24k/normalized_txt.flist \
      ${dumpdir}/${name}/phoneme
  done

  for name in test-clean test-other; do
    echo "Resample ${name} set to ${dumpdir}/${name}/wavs/*.wav"
    torchrun --nproc_per_node=32 --master_port=1234 scripts/convert_to_wav.py \
      --wav_scp ${dumpdir}/${name}_24k/wav.scp \
      --out_dir ${dumpdir}/${name}/wavs \
      --sample_rate 16000

    mkdir -p ${dumpdir}/${name}
    find ${dumpdir}/${name}/wavs/ -iname "*.wav" | awk -F '/' '{print $NF, $0}' | sort >> ${dumpdir}/${name}/wav.scp

    echo "Collect and tokenize text files of ${name} into one phoneme file"
    python scripts/collect_text_flist_to_phone_scp.py \
      ${dumpdir}/${name}_24k/nomalized_txt.flist \
      ${dumpdir}/${name}/phoneme
  done
fi

# extract codec
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "stage 4: Extract codecs"
  home_dir=$(pwd)
  cd ../codec
  for name in train dev test-clean test-other; do
    echo "extracting codec for ${name}"
    sh encoding_decoding.sh --stage 1 \
      --gpu_devices ${gpu_devices} \
      --njob ${njob} \
      --bit_width 32000 \
      --batch_size 8 \
      --data_format kaldi_ark \
      --indices_save_type ark \
      --model_dir "${home_dir}/exp/${codec_model}" \
      --wav_scp "${home_dir}/${dumpdir}/${name}/wav.scp" \
      --out_dir "${home_dir}/${dumpdir}/${name}/codecs/"

    cat ${home_dir}/${dumpdir}/${name}/codecs/logdir/output.*/indices.scp | sort > ${home_dir}/${dumpdir}/${name}/codec_tokens.scp
    echo "codec scp files are collected into ${home_dir}/${dumpdir}/${name}/codec_tokens.scp"
  done
fi

# stage 5: training model
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "stage 5: training"
  mkdir -p ${exp_dir}/exp/${model_dir}
  mkdir -p ${exp_dir}/exp/${model_dir}/log
  INIT_FILE=${exp_dir}/exp/${model_dir}/ddp_init
  if [ -f $INIT_FILE ];then
      rm -f $INIT_FILE
  fi
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  for ((i = 0; i < ${ngpu}; ++i)); do
      {
          rank=$i
          local_rank=$i
          gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
          python -m funasr.bin.text2audio_train \
              --gpu_id $gpu_id \
              --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/train/phoneme,text,text \
              --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/train/codec_token.scp,codec,kaldi_ark \
              --train_shape_file ${feats_dir}/${state_dir}/train/codec_shape \
              --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/dev/phoneme,text,text \
              --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/dev/codec_token.scp,codec,kaldi_ark \
              --valid_shape_file ${feats_dir}/${state_dir}/dev/codec_shape \
              --init_param exp/${codec_model}/model.pth:quantizer.rq.model:quantizer_codebook exp/${codec_model}/model.pth:quantizer:quantizer \
              --token_list data/en_phoneme_token.list \
              --token_type word \
              --ignore_init_mismatch true \
              --resume true \
              --output_dir ${exp_dir}/exp/${model_dir} \
              --config $train_config \
              --ngpu ${ngpu} \
              --num_worker_count 1 \
              --multiprocessing_distributed true \
              --dist_init_method $init_method \
              --dist_world_size $ngpu \
              --dist_rank $rank \
              --local_rank $local_rank 1> ${exp_dir}/exp/${model_dir}/log/train.log.$i 2>&1
      } &
      done
      echo "log files are "${exp_dir}/exp/${model_dir}/log/train.log.*
      wait
fi

# stage 6: inference
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: inference"

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
