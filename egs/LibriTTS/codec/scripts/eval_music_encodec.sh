#!/usr/bin/env bash

# . ./path.sh || exit 1;

# general configuration
feats_dir="." #feature output dictionary
exp_dir="exp"
nq=2
dumpdir=dump/ptts_16k
stage=4
stop_stage=7
infer_cmd=utils/run.pl
test_sets=""
njob=1
docker_nj=32
gpu_inference=true
gpu_devices="0,1"
eval_tag="music_encodec_32khz"

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

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

# Testing Stage
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Inference with nq=${nq}"
    for dset in ${test_sets}; do
        echo "Encoding for $dset nq=${nq}"
        asr_exp=${exp_dir}/${eval_tag}
        _dir="${asr_exp}/inference_nq${nq}/${dset}"
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
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}
        # add dummy "0,", since JOB start from 1 rather than 0.
        ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/encode.JOB.log \
            CUDA_VISIBLE_DEVICES=0,${gpu_devices} python scripts/eval_music_encodec.py \
              --nq ${nq} \
              --in_scp "${_logdir}"/keys.JOB.scp \
              --out_dir "${_logdir}"/output.JOB \
              --device cuda:JOB
    done
fi

nj=${docker_nj}
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "stage 5: Prepare test files with nq=${nq}"
  for dset in ${test_sets}; do
    echo "Processing for $dset"
    _data="${feats_dir}/${dumpdir}/${dset}"
    asr_exp=${exp_dir}/${eval_tag}
    _dir="${asr_exp}/inference_nq${nq}/${dset}"
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
  echo "stage 6: Scoring with nq=${nq}"
  for dset in ${test_sets}; do
    asr_exp=${exp_dir}/${eval_tag}
    out_dir="${asr_exp}/inference_nq${nq}/${dset}"
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
  echo "stage 7: Calculating averaged ViSqol scores with nq=${nq}"
  for dset in ${test_sets}; do
    asr_exp=${exp_dir}/${eval_tag}
    out_dir="${asr_exp}/inference_nq${nq}/${dset}"
    cat ${out_dir}/split${nj}/*.res | grep -v "reference,degraded,moslqo" > ${out_dir}/visqol.res
    avg_score=`awk -F',' 'BEGIN{score=0.0}{score+=$3;}END{print score/NR;}' ${out_dir}/visqol.res`
    echo "Average ViSqol: ${avg_score}" | tee ${out_dir}/result
  done
fi
