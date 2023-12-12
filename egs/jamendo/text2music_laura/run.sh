#!/usr/bin/env bash

. ./path.sh || exit 1;

# global configs
stage=1
stop_stage=1

# data related
corpus_dir=corpus/Jamendo
dumpdir=dump/jamendo
state_dir=exp/jamendo_states

# pre-trained related
model_name=
model_hub=modelscope
codec_model="audio_codec-freqcodec-universal-general-16k-nq32ds640-pytorch"

# training related
train_config="conf/text2audio_codec_lm_nq2_uni_rel_pos.yaml"
tag="_t5_enc_jamendo"
feats_dir="."
exp_dir="."

# inference related
text_scp=text
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
  cd ${corpus_dir}
  git clone https://github.com/MTG/mtg-jamendo-dataset.git
  cd mtg-jamendo-dataset
  pip install -r scripts/requirements.txt
  python3 scripts/download/download.py \
    --dataset raw_30s \
    --type audio --unpack --remove \
    ./Jamendo
fi

# Data collecting and conversion
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: convert mp3 files into waveform files."
  data_dir=${dumpdir}/all
  mkdir -p ${data_dir}
  find ${corpus_dir}/mtg-jamendo-dataset/Jamendo -iname "*.mp3" | sort > ${data_dir}/mp3.flist
  while IFS= read -r line; do
    mp3_file=$(echo "$line" | awk '{print $2}')
    wav_file=$(echo "${mp3_file}" | sed "s:.mp3:.wav:g")
    ffmpeg -i "${mp3_file}" -acodec pcm_s16le -ac 1 -ar 16000 -nostdin "${wav_file}" 1>> ${data_dir}/convert.log 2>&1
  done < "${data_dir}/mp3.flist"

  find ${corpus_dir}/mtg-jamendo-dataset/Jamendo -iname "*.wav" | sort | awk -F'[/.]' '{print $(NF-1),$0}' > ${data_dir}/reco.scp

  head -n1 ${corpus_dir}/mtg-jamendo-dataset/data/raw_30s.tsv >> ${data_dir}/tag.tsv
  cat ${corpus_dir}/mtg-jamendo-dataset/data/raw_30s.tsv | grep genre | grep instrument | grep "mood/theme" >> ${data_dir}/tag.tsv
  python scripts/preprocess_jamendo_tsv.py \
    --tsv_file ${data_dir}/tag.tsv \
    --out_file ${data_dir}/tag.scp
fi

# data preprocess
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: Data preprocessing"
  echo "Stage 3-1: clip recordings into 10s segments"
  data_dir=${dumpdir}/all
  torchrun --nproc_per_node=32 --master_port=62211 \
    scripts/clip_audio_to_seg.py \
    --wav_scp ${data_dir}/reco.scp \
    --seg_dur 10.0 \
    --out_dir ${corpus_dir}/mtg-jamendo-dataset/Jamendo/clips_10s
  cat ${corpus_dir}/mtg-jamendo-dataset/Jamendo/clips_10s/part*.scp | sort > ${data_dir}/all_wav_clips.scp

  echo "Stage 3-2: filtering out segments without full tag"
  python scripts/filter_wav_by_tag_scp.py \
    --wav_scp ${data_dir}/all_wav_clips.scp \
    --tag_scp ${data_dir}/tag.scp \
    --out_dir ${data_dir}

  echo "Stage 3-3: split train, dev and test subsets"
  awk '{print $1}' ${data_dir}/wav.scp | shuf > ${data_dir}/uttids

  mkdir -p ${dumpdir}/{train,dev,test}

  total_lines=$(cat ${dumpdir}/all/uttids | wc -l)
  head -n $(( total_lines - 2000 )) ${dumpdir}/all/uttids | sort > ${dumpdir}/train/uttids
  tail -n 2000 ${dumpdir}/all/uttids | head -n 1000 | sort > ${dumpdir}/dev/uttids
  tail -n 1000 ${dumpdir}/all/uttids | sort > ${dumpdir}/test/uttids
  for name in train dev test; do
    awk '{if (NR==FNR){a[$1]=1}else{if (a[$1]==1){print $0}}}' ${dumpdir}/${name}/uttids ${dumpdir}/all/wav.scp > ${dumpdir}/${name}/wav.scp
    awk '{if (NR==FNR){a[$1]=1}else{if (a[$1]==1){print $0}}}' ${dumpdir}/${name}/uttids ${dumpdir}/all/text > ${dumpdir}/${name}/text
  done
fi

# extract codec
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "stage 4: Extract codecs"
  home_dir=$(pwd)
  cd ../../LibriTTS/codec
  for name in train dev test; do
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
    cat ${home_dir}/${dumpdir}/${name}/codecs/logdir/output.*/indices.scp | sort > ${home_dir}/${dumpdir}/${name}/codec_token.scp
    echo "codec scp files are collected into ${home_dir}/${dumpdir}/${name}/codec_token.scp"

    mkdir -p "${home_dir}/${state_dir}/${name}"
    awk '{print $1,250}' ${home_dir}/${dumpdir}/${name}/codec_token.scp > ${home_dir}/${state_dir}/${name}/codec_shape
  done

  for name in train dev test; do
    echo "extracting text embeddings for ${name}"
    torchrun --nproc_per_node=${inference_nj} --master_port=62211 \
      --text "${home_dir}/${dumpdir}/${name}/text" \
      --gpu_list ${gpu_devices} \
      --nlp_model "./exp/t5-base" \
      --emb_type "enc" \
      --out_dir "${home_dir}/${dumpdir}/${name}/t5_embeddings"

    cat ${home_dir}/${dumpdir}/${name}/t5_embeddings/part*.scp | sort > ${home_dir}/${dumpdir}/${name}/text_emb.scp
    echo "text embeddings are collected into ${home_dir}/${dumpdir}/${name}/text_emb.scp"
  done
fi

# stage 5: training model
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "stage 5: training"
  mkdir -p ${exp_dir}/${model_dir}
  mkdir -p ${exp_dir}/${model_dir}/log
  INIT_FILE=${exp_dir}/${model_dir}/ddp_init
  if [ -f $INIT_FILE ];then
      rm -f $INIT_FILE
  fi
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  for ((i = 0; i < ${ngpu}; ++i)); do
      {
          rank=$i
          local_rank=$i
          gpu_id=$(echo $gpu_devices | cut -d',' -f$[$i+1])
          python -m funcodec.bin.text2audio_train \
              --gpu_id $gpu_id \
              --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/train/text_emb.scp,text,kaldi_ark \
              --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/train/codec_token.scp,codec,kaldi_ark \
              --train_shape_file ${feats_dir}/${state_dir}/train/codec_shape \
              --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/dev/text_emb.scp,text,kaldi_ark \
              --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/dev/codec_token.scp,codec,kaldi_ark \
              --valid_shape_file ${feats_dir}/${state_dir}/dev/codec_shape \
              --init_param exp/${codec_model}/model.pth:quantizer.rq.model:quantizer_codebook exp/${codec_model}/model.pth:quantizer:quantizer \
              --ignore_init_mismatch true \
              --resume true \
              --output_dir ${exp_dir}/${model_dir} \
              --config $train_config \
              --ngpu ${ngpu} \
              --num_worker_count 1 \
              --multiprocessing_distributed true \
              --dist_init_method $init_method \
              --dist_world_size $ngpu \
              --dist_rank $rank \
              --local_rank $local_rank 1> ${exp_dir}/${model_dir}/log/train.log.$i 2>&1
      } &
      done
      echo "log files are "${exp_dir}/${model_dir}/log/train.log.*
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
            --text_emb_model exp/t5-base

    echo "Generated audios are saved to ${_logdir}/output.*/*.wav"

fi
