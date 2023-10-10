#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
gpu_devices="6,7"
gpu_num=2
count=1

# general configuration
feats_dir="." #feature output dictionary
exp_dir="."
dumpdir=dump/LibriTTS
stage=0
stop_stage=3
corpus_dir=corpus/LibriTTS

# training related
tag=""
train_set=train
valid_set=dev
train_config=conf/encodec_lstm_16k_n32_600k_step_rmseg.yaml
init_param=
state_dir=LibriTTS_states

# inference related
inference_model=30epoch.pth
inference_tag="inference"
batch_size=1
test_sets="test-clean"
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding
need_indices=false
need_sub_quants=false
# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=5
docker_nj=32
infer_cmd=utils/run.pl
sample_frequency=16000
file_sampling_rate=16000
bit_width=4000
use_scale=false
use_ppg=false
model_dir=

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ -z "${model_dir}" ]; then
  model_dir="$(basename "${train_config}" .yaml)_${tag}"
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

# Data downloading
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0: data downloading"

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
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: collecting data sets."
  mkdir -p ${dumpdir}/train_24k ${dumpdir}/dev_24k

  for name in train-clean-100 train-clean-360 train-other-500; do
    echo "collecting ${name} in to ${dumpdir}/train_24k/wav.scp"
    find ${corpus_dir}/${name}/ -iname "*.wav" | awk -F '/' '{print $NF, $0}' | sort >> ${dumpdir}/train_24k/wav.scp
  done

  for name in dev-clean dev-other; do
    echo "collecting ${name} in to ${dumpdir}/dev_24k/wav.scp"
    find ${corpus_dir}/${name}/ -iname "*.wav" | awk -F '/' '{print $NF, $0}' | sort >> ${dumpdir}/dev_24k/wav.scp
  done

  for name in test-clean test-other; do
    mkdir -p ${dumpdir}/${name}_24k
    echo "collecting ${name} in to ${dumpdir}/${name}_24k/wav.scp"
    find ${corpus_dir}/${name}/ -iname "*.wav" | awk -F '/' '{print $NF, $0}' | sort > ${dumpdir}/${name}_24k/wav.scp
  done
fi

# Dump data to ark and convert it to the sampling rate of 16000
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: Dump data to ark."
  for name in train dev; do
    echo "Dump ${name} set to ark files ${dumpdir}/${name}/arks/wav.*.ark"
    torchrun --nproc_per_node=32 --master_port=1234 scripts/preprocess_and_dump.py \
      --wav_scp ${dumpdir}/${name}_24k/wav.scp \
      --out_dir ${dumpdir}/${name}/arks \
      --sample_rate 16000

    mkdir -p ${dumpdir}/${name} exp/${state_dir}/${name}
    cat ${dumpdir}/${name}/arks/wav.*.scp | sort > ${dumpdir}/${name}/wav.scp
    cat ${dumpdir}/${name}/arks/length.*.txt | shuf > exp/${state_dir}/${name}/speech_shape
  done

  for name in test-clean test-other; do
    echo "Resample ${name} set to ${dumpdir}/${name}/wavs/*.wav"
    torchrun --nproc_per_node=32 --master_port=1234 scripts/convert_to_wav.py \
      --wav_scp ${dumpdir}/${name}_24k/wav.scp \
      --out_dir ${dumpdir}/${name}/wavs \
      --sample_rate 16000

    mkdir -p ${dumpdir}/${name}
    find ${dumpdir}/${name}/wavs/ -iname "*.wav" | awk -F '/' '{print $NF, $0}' | sort >> ${dumpdir}/${name}/wav.scp
  done
fi

# Training Stage
world_size=$gpu_num  # run on one machine
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Training"
    mkdir -p ${exp_dir}/exp/${model_dir}
    mkdir -p ${exp_dir}/exp/${model_dir}/log
    INIT_FILE=${exp_dir}/exp/${model_dir}/ddp_init
    if [ -f $INIT_FILE ];then
        rm -f $INIT_FILE
    fi
    ppg_opt=""
    if [ ${use_ppg} == true ]; then
      ppg_opt="
      --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${train_set}/ppg.scp,ppg,kaldi_ark
      --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${valid_set}/ppg.scp,ppg,kaldi_ark
      "
    fi
    init_opt=""
    if [ ! -z "${init_param}" ]; then
        init_opt="--init_param ${init_param}"
        echo ${init_opt}
    fi

    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    for ((i = 0; i < $gpu_num; ++i)); do
        {
            rank=$i
            local_rank=$i
            gpu_id=$(echo $gpu_devices | cut -d',' -f$[$i+1])
            python -m funcodec.bin.codec_train \
                --gpu_id $gpu_id \
                --use_preprocessor true \
                --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${train_set}/wav.scp,speech,kaldi_ark \
                --train_shape_file ${feats_dir}/exp/${state_dir}/${train_set}/speech_shape \
                --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${valid_set}/wav.scp,speech,kaldi_ark \
                --valid_shape_file ${feats_dir}/exp/${state_dir}/${valid_set}/speech_shape \
                ${init_opt} --ignore_init_mismatch true \
                ${ppg_opt} --resume true \
                --output_dir ${exp_dir}/exp/${model_dir} \
                --config $train_config \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --multiprocessing_distributed true \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${exp_dir}/exp/${model_dir}/log/train.log.$i 2>&1
        } &
        done
        wait
fi

# Testing Stage
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Inference @ ${bit_width}"
    for dset in ${test_sets}; do
        echo "Processing for $dset @ ${bit_width}"
        asr_exp=${exp_dir}/exp/${model_dir}
        _dir="${asr_exp}/${inference_tag}/${inference_model}/${dset}"
        _logdir="${_dir}/logdir"
        if [ -d ${_dir} ]; then
            echo "WARNING: ${_dir} is already exists."
            exit 0
        fi
        mkdir -p "${_logdir}"
        _data="${feats_dir}/${dumpdir}/${dset}"
        key_file=${_data}/wav.scp
        num_scp_file="$(<${key_file} wc -l)"
        _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
        split_scps=
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        ppg_opt=""
        if [ ${use_ppg} == true ]; then
          ppg_opt="--data_path_and_name_and_type ${_data}/ppg.scp,ppg,kaldi_ark"
        fi
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}
        ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/inference.JOB.log \
            python -m funcodec.bin.codec_inference \
                --batch_size ${batch_size} \
                --ngpu "${_ngpu}" \
                --gpuid_list ${gpuid_list} \
                --data_path_and_name_and_type "${_data}/wav.scp,speech,sound" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --config_file "${asr_exp}"/config.yaml \
                --model_file "${asr_exp}"/"${inference_model}" \
                --output_dir "${_logdir}"/output.JOB \
                --sampling_rate $sample_frequency \
                --file_sampling_rate $file_sampling_rate \
                --bit_width ${bit_width} \
                --need_indices ${need_indices} \
                --need_sub_quants ${need_sub_quants} \
                --use_scale ${use_scale} ${ppg_opt}
    done
fi

nj=${docker_nj}
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "stage 5: Prepare test files @ ${bit_width}"
  for dset in ${test_sets}; do
    echo "Processing for $dset"
    asr_exp=${exp_dir}/exp/${model_dir}
    _data="${feats_dir}/${dumpdir}/${dset}"
    _dir="${asr_exp}/${inference_tag}/${inference_model}/${dset}"
    _logdir="${_dir}/logdir"
    find `pwd`/${_logdir}/ -iname "*.wav" | awk -F'[/.]' '{print $(NF-1),$0}' | sort > ${_dir}/degrad.scp
    awk '{print $2}' ${_dir}/degrad.scp > ${_dir}/degrad.flist

    cat ${_data}/wav.scp | sort | awk '{print $2}' > ${_dir}/ref.flist
    paste -d "," ${_dir}/ref.flist ${_dir}/degrad.flist > ${_dir}/ref_hyp.flist
    mkdir -p ${_dir}/split${nj}
    split -l$((`wc -l < ${_dir}/ref_hyp.flist`/${nj})) --numeric-suffixes \
      ${_dir}/ref_hyp.flist ${_dir}/split${nj}/part.
    for name in `ls ${_dir}/split${nj}`; do
      sed -i '1 i\reference,degraded' ${_dir}/split${nj}/${name};
    done
  done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "stage 6: Scoring @ ${bit_width}"
  for dset in ${test_sets}; do
    asr_exp=${exp_dir}/exp/${model_dir}
    out_dir="${asr_exp}/${inference_tag}/${inference_model}/${dset}"
    for name in `ls ${out_dir}/split${nj}/part.*`; do
      docker run -t -v /nfs:/nfs -v /home:/home jonashaag/visqol:v3 \
        -batch_input_csv `pwd`/${name} \
        -results_csv `pwd`/${name}.res 2>&1 > /dev/null &
    done
    echo "Waiting for calculation..."
    wait
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "stage 7: Calculating averaged ViSqol scores @ ${bit_width}"
  for dset in ${test_sets}; do
    asr_exp=${exp_dir}/exp/${model_dir}
    out_dir="${asr_exp}/${inference_tag}/${inference_model}/${dset}"
    cat ${out_dir}/split${nj}/*.res | grep -v "reference,degraded,moslqo" > ${out_dir}/visqol.res
    avg_score=`awk -F',' 'BEGIN{score=0.0}{score+=$3;}END{print score/NR;}' ${out_dir}/visqol.res`
    echo "Average ViSqol: ${avg_score}" | tee ${out_dir}/result
  done
fi

# Training Stage
world_size=$gpu_num  # run on one machine
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Training large dataloader"
    mkdir -p ${exp_dir}/exp/${model_dir}
    mkdir -p ${exp_dir}/exp/${model_dir}/log
    INIT_FILE=${exp_dir}/exp/${model_dir}/ddp_init
    if [ -f $INIT_FILE ];then
        rm -f $INIT_FILE
    fi
    init_opt=""
    if [ ! -z "${init_param}" ]; then
        init_opt="--init_param ${init_param}"
        echo ${init_opt}
    fi

    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    for ((i = 0; i < $gpu_num; ++i)); do
        {
            rank=$i
            local_rank=$i
            gpu_id=$(echo $gpu_devices | cut -d',' -f$[$i+1])
            python -m funcodec.bin.codec_train \
                --gpu_id $gpu_id \
                --use_preprocessor false \
                --dataset_type large \
                --train_data_file ${feats_dir}/${dumpdir}/${train_set}/data_file.list \
                --valid_data_file ${feats_dir}/${dumpdir}/${valid_set}/data_file.list \
                ${init_opt} --ignore_init_mismatch true \
                --allow_variable_data_keys true \
                --resume true \
                --output_dir ${exp_dir}/exp/${model_dir} \
                --config $train_config \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --multiprocessing_distributed true \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${exp_dir}/exp/${model_dir}/log/train.log.$i 2>&1
        } &
        done

        echo "log can be found at ${exp_dir}/exp/${model_dir}/log/train.log.*"
        wait
fi