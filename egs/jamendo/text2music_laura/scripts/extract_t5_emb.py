"""
Author: Zhihao Du
Date: 2023.03.29
Description: Convert wav, flac and ark files to mono waveform files with given sampling rate.
"""
import logging
import warnings
warnings.filterwarnings("ignore")
import os
import time
import argparse
import torch
from funasr.utils.misc import get_logger
import kaldiio


def main(args):
    logger = get_logger(log_level=logging.INFO)
    rank = int(os.environ['LOCAL_RANK'])
    threads_num = int(os.environ['WORLD_SIZE'])
    out_dir = args.out_dir
    local_device = "cpu"
    if len(args.gpu_list) > 0:
        local_device = f"cuda:{rank % len(args.gpu_list)}"

    def local_logging(msg: str):
        logger.info("rank {}/{}: {}".format(
            rank, threads_num, msg
        ))

    local_logging("nlp_model: {}, local_device: {}, out_dir {}".format(
        args.nlp_model, local_device, out_dir
    ))
    if out_dir is not None:
        if rank == 0:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        else:
            while not os.path.exists(out_dir):
                time.sleep(0.5)

    local_logging("loading data...")
    all_recs = []
    for one_line in open(args.text, "rt", encoding="utf-8"):
        key, path = one_line.strip().split(maxsplit=1)
        all_recs.append((key, path))
    all_recs.sort(key=lambda x: x[0])
    local_all_recs = all_recs[rank::threads_num]
    local_logging("done.")

    local_logging("loading nlp model...")
    from transformers import T5Tokenizer, T5Model
    tokenizer = T5Tokenizer.from_pretrained(args.nlp_model)
    model = T5Model.from_pretrained(args.nlp_model)
    model = model.to(local_device)
    local_logging("done.")

    out_path = os.path.join(out_dir, f"part{rank:02d}")
    ark_writer = kaldiio.WriteHelper(f"ark,scp,f:{out_path}.ark,{out_path}.scp")
    meeting_count = 0
    for i, (uttid, text) in enumerate(local_all_recs):
        # input_ids = tokenizer("Hi, little dog.", return_tensors="pt").input_ids
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = input_ids.to(local_device)
        with torch.no_grad():
            if args.emb_type == "enc":
                    outputs = model.encoder(input_ids).last_hidden_state
            else:
                outputs = model.shared(input_ids)

        outputs = outputs.cpu().squeeze(0).numpy()

        ark_writer(uttid, outputs)
        if i % 100 == 0:
            local_logging(f"{(i+1)/len(local_all_recs)*100.0:.2f}%: process {uttid}")

        meeting_count += 1

    ark_writer.close()
    local_logging("Complete {} records".format(meeting_count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text',
                        type=str,
                        default=None,
                        help="kaldi-style wav path script")
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help="local rank of gpu")
    parser.add_argument('--gpu_list',
                        type=str,
                        default="",
                        help="the list of gpu devices.")
    parser.add_argument('--nlp_model',
                        type=str,
                        default="./bloom_1b1/",
                        help="The nlp model used to extract embeddings.")
    parser.add_argument('--emb_type',
                        type=str,
                        default="emb",
                        choices=["emb", "enc"],
                        help="The embedding type to extract.")
    parser.add_argument('--out_dir',
                        type=str,
                        default=None,
                        help="The output dir to save rttms and wavs")
    args = parser.parse_args()
    args.gpu_list = args.gpu_list.split(",")
    main(args)
