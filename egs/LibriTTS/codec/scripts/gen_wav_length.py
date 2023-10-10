"""
Author: Zhihao Du
Date: 2023.03.29
Description: Collect the length of files in wav, flac and ark files.
"""
import logging
import warnings
warnings.filterwarnings("ignore")
import os
import time
import argparse
import numpy as np
import kaldiio
import librosa
import soundfile
from tqdm import tqdm


def get_logger(fpath=None, log_level=logging.INFO):
    formatter = logging.Formatter(
        f"[{os.uname()[1].split('.')[0]}]"
        f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    logging.basicConfig(
        level=log_level,
        format=f"[{os.uname()[1].split('.')[0]}]"
               f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("Pyobj, f")
    if fpath is not None:
        # Dump log to file
        fh = logging.FileHandler(fpath)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def main(args):
    logger = get_logger(log_level=logging.INFO)
    rank = int(os.environ['LOCAL_RANK'])
    threads_num = int(os.environ['WORLD_SIZE'])
    out_dir = args.out_dir
    logger.info("rank {}/{}: out_dir {}.".format(
        rank, threads_num, out_dir
    ))
    if out_dir is not None:
        if rank == 0:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        else:
            while not os.path.exists(out_dir):
                time.sleep(0.5)

    all_recs = []
    if args.wav_scp is not None and len(args.wav_scp) > 0:
        for one_line in open(args.wav_scp, "rt", encoding="utf-8"):
            path = one_line.strip()
            key, path = path.split(" ", maxsplit=1)
            all_recs.append((key, path))
    else:
        for one_line in open(args.wav_list, "rt", encoding="utf-8"):
            path = one_line.strip()
            key = os.path.basename(path).rsplit(".", 1)[0]
            all_recs.append((key, path))
    all_recs.sort(key=lambda x: x[0])
    local_all_recs = all_recs[rank::threads_num]

    length_file = open(os.path.join(out_dir, f"wav_length.{rank}.txt"), "wt")
    meeting_count = 0
    for i, (uttid, wav_path) in tqdm(enumerate(local_all_recs), total=len(local_all_recs), ascii=True):
        retval = kaldiio.load_mat(wav_path)
        if isinstance(retval, tuple):
            assert len(retval) == 2, len(retval)
            if isinstance(retval[0], int) and isinstance(retval[1], np.ndarray):
                # sound scp case
                rate, array = retval
            elif isinstance(retval[1], int) and isinstance(retval[0], np.ndarray):
                # Extended ark format case
                array, rate = retval
            else:
                raise RuntimeError(
                    f"Unexpected type: {type(retval[0])}, {type(retval[1])}"
                )
        else:
            # Normal ark case
            assert isinstance(retval, np.ndarray), type(retval)
            array = retval

        length_file.write(f"{uttid} {array.shape[0]}\n")

        if i % 100 == 0:
            logger.info("{}/{}: process {}/{}: {}.".format(rank, threads_num, i, len(local_all_recs), uttid))
            length_file.flush()

        meeting_count += 1

    logger.info("{}/{}: Complete {} records.".format(rank, threads_num, meeting_count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_list',
                        type=str,
                        default=None,
                        help="wav path list")
    parser.add_argument('--wav_scp',
                        type=str,
                        default=None,
                        help="kaldi-style wav path script")
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help="local rank of gpu")
    parser.add_argument('--out_dir',
                        type=str,
                        default=None,
                        help="The output dir to save rttms and wavs")
    args = parser.parse_args()
    main(args)
