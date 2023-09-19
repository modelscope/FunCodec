"""
Author: Zhihao Du
Date: 2023.03.29
Description: Convert phoneme lab files to one-hot ppg file.
"""
import logging
import warnings
warnings.filterwarnings("ignore")
import os
import time
import argparse
import numpy as np
from funcodec.utils.misc import get_logger
import kaldiio
import textgrid


def read_lab_as_ppg_idx(lab_path, dict_list, frame_shift):
    text_grid = []
    dur = 0
    with open(lab_path, "rt") as fd:
        for one_line in fd:
            parts = one_line.strip().split("\t")
            st, ed = int(float(parts[0]) / frame_shift), int(float(parts[1]) / frame_shift)
            if len(parts) == 2 or parts[2] == "" or len(parts[2]) == 0 or parts[2] is None:
                idx = dict_list.index("sil")
            else:
                if parts[2] in dict_list:
                    idx = dict_list.index(parts[2])
                else:
                    idx = dict_list.index("OOV")
            text_grid.append((st, ed, idx))
            dur = max(float(parts[1]), dur)
    align = np.zeros((int(dur/frame_shift),), dtype=np.int32)
    for st, ed, idx in text_grid:
        align[st: ed] = idx

    return align


def read_textgrid_as_ppg_idx(textgrid_path, dict_list, frame_shift):
    text_grid = []
    dur = 0
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_path)
    except ValueError:
        tg = None
    if tg is None:
        return None
    for tier in tg:
        if tier.name != "phones":
            continue
        for i, seg in enumerate(tier):
            st, ed = int(float(seg.minTime) / frame_shift), int(float(seg.maxTime) / frame_shift)
            if seg.mark == "" or len(seg.mark) == 0 or seg.mark is None:
                idx = dict_list.index("sil")
            else:
                if seg.mark in dict_list:
                    idx = dict_list.index(seg.mark)
                else:
                    idx = dict_list.index("OOV")
            text_grid.append((st, ed, idx))
            dur = max(float(seg.maxTime), dur)
    align = np.zeros((int(dur/frame_shift),), dtype=np.int32)
    for st, ed, idx in text_grid:
        align[st: ed] = idx

    return align


def main(args):
    logger = get_logger(log_level=logging.INFO)
    rank = int(os.environ['LOCAL_RANK'])
    threads_num = int(os.environ['WORLD_SIZE'])
    out_dir = args.out_dir
    logger.info("rank {}/{}: start, out_dir {}.".format(
        rank, threads_num, out_dir
    ))
    if out_dir is not None:
        if rank == 0:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        else:
            while not os.path.exists(out_dir):
                time.sleep(0.5)

    token_list = []
    with open(args.dict_path, "rt") as fd:
        for one_line in fd:
            token_list.append(one_line.strip())

    all_recs = []
    if args.lab_scp is not None and len(args.lab_scp) > 0:
        for one_line in open(args.lab_scp, "rt", encoding="utf-8"):
            path = one_line.strip()
            key, path = path.split(" ", maxsplit=1)
            all_recs.append((key, path))
    else:
        for one_line in open(args.lab_list, "rt", encoding="utf-8"):
            path = one_line.strip()
            key = os.path.basename(path).rsplit(".", 1)[0]
            all_recs.append((key, path))
    all_recs.sort(key=lambda x: x[0])
    local_all_recs = all_recs[rank::threads_num]

    count = 0
    outfile_path = os.path.join(out_dir, "lab.{}".format(rank))
    ppg_writer = kaldiio.WriteHelper("ark,scp,f:{}.ark,{}.scp".format(outfile_path, outfile_path))
    for i, (uttid, lab_path) in enumerate(local_all_recs):
        # float matrix in the shape of Tx1
        if lab_path.endswith(".TextGrid"):
            ppg_idx = read_textgrid_as_ppg_idx(lab_path, token_list, args.frame_shift)
        else:
            ppg_idx = read_lab_as_ppg_idx(lab_path, token_list, args.frame_shift)
        if ppg_idx is not None:
            ppg_writer(uttid, ppg_idx)
        else:
            logger.info(f"{rank}/{threads_num}: fail to parse {uttid}, skip it.")

        if i % 100 == 0:
            logger.info("{}/{}: process {}.".format(rank, threads_num, uttid))

        count += 1

    logger.info("{}/{}: Complete {} records.".format(rank, threads_num, count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_path',
                        type=str,
                        default=None,
                        help="path to dictionary")
    parser.add_argument('--lab_list',
                        type=str,
                        default=None,
                        help="lab path list")
    parser.add_argument('--lab_scp',
                        type=str,
                        default=None,
                        help="kaldi-style lab path script")
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help="local rank of gpu")
    parser.add_argument('--out_dir',
                        type=str,
                        default=None,
                        help="The output dir to save rttms and wavs")
    parser.add_argument('--frame_shift',
                        type=float,
                        default=0.01,
                        help="The expected sample rate.")
    args = parser.parse_args()
    main(args)
