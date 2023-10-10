"""
Author: Zhihao Du
Date: 2023.03.29
Description: Preprocess waveform files and dump them to ark file for training
- Filter out not-wav files
- Resample rate to 16k
- Dump data to ark files
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
import librosa


def main(args):
    logger = get_logger(log_level=logging.INFO)
    rank = int(os.environ['LOCAL_RANK'])
    threads_num = int(os.environ['WORLD_SIZE'])
    sr, sample_bits = args.sample_rate, 16
    out_dir = args.out_dir
    logger.info("rank {}/{}: Sample rate {}, sample bits {}, out_dir {}.".format(
        rank, threads_num, sr, sample_bits, out_dir
    ))
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

    outfile_path = os.path.join(out_dir, "wav.{}".format(rank))
    wav_writer = kaldiio.WriteHelper("ark,scp,f:{}.ark,{}.scp".format(outfile_path, outfile_path))
    outfile_path = os.path.join(out_dir, "length.{}.txt".format(rank))
    length_writer = open(outfile_path, "wt")

    meeting_count = 0
    for i, (uttid, wav_path) in enumerate(local_all_recs):
        # skip files not ending with wav
        if not wav_path.lower().endswith(".wav"):
            logger.warning("rank {}/{}: Skip {} since {} file format is not wav.".format(
                rank, threads_num, uttid, wav_path
            ))
            continue
        # Use librosa to deal with multi-channels and different sampling rate
        wav, sr = librosa.load(wav_path, dtype=np.float32, sr=sr, mono=True)
        wav_writer(uttid, wav)
        length_writer.write("{} {}\n".format(uttid, len(wav)))

        if i % 100 == 0:
            logger.info("{}/{}: process {}.".format(rank, threads_num, uttid))

        meeting_count += 1

    wav_writer.close()
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
                        default="",
                        help="The output dir to save rttms and wavs")
    parser.add_argument('--sample_rate',
                        type=int,
                        default=16_000,
                        help="The expected sample rate.")
    args = parser.parse_args()
    main(args)
