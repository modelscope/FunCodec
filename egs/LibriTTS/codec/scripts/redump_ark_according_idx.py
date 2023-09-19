"""
Author: Zhihao Du
Date: 2023.03.29
Description: Redump ark file with the sort in uttidx file
"""
import logging
import os
logging.basicConfig(
        level=logging.INFO,
        format=f"[{os.uname()[1].split('.')[0]}]"
               f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
import warnings
warnings.filterwarnings("ignore")
import time
import argparse
import kaldiio
import numpy as np


def main(args):
    if args.use_arg:
        rank = args.local_rank
        threads_num = args.world_size
    else:
        rank = int(os.environ['LOCAL_RANK'])
        threads_num = int(os.environ['WORLD_SIZE'])
    sr, sample_bits = args.sample_rate, 16
    out_dir = args.out_dir
    logging.info("rank {}/{}: Sample rate {}, sample bits {}, out_dir {}.".format(
        rank, threads_num, sr, sample_bits, out_dir
    ))
    if rank == 0:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    else:
        while not os.path.exists(out_dir):
            time.sleep(0.5)

    ark_file_list = set()
    ark_scp = {}
    for one_line in open(args.ark_scp, "rt", encoding="utf-8"):
        path = one_line.strip()
        key, path = path.split(maxsplit=1)
        ark_scp[key] = path
        ark_file_path = path.split(":")[0]
        ark_file_list.add(ark_file_path)
    ark_file_list = list(ark_file_list)
    logging.info(f"Found {len(ark_file_list)} ark files.")
    all_exits = True
    for one in ark_file_list:
        if not os.path.exists(one):
            logging.info(f"{one} not exists")
            all_exits = False
    if not all_exits:
        logging.info(f"Not all ark file exists, exit.")
        return

    all_recs = []
    for one_line in open(args.uttidx, "rt", encoding="utf-8"):
        all_recs.append(one_line.strip().split(maxsplit=1)[0])
    local_all_recs = all_recs[rank::threads_num]

    ark_name = os.path.basename(args.ark_scp).split(".", maxsplit=1)[0]
    outfile_path = os.path.join(out_dir, f"{ark_name}_part{rank:02d}")
    ark_writer = kaldiio.WriteHelper("ark,scp:{}.ark,{}.scp".format(outfile_path, outfile_path))
    outfile_path = os.path.join(out_dir, f"{ark_name}_part{rank:02d}_length.txt")
    length_writer = open(outfile_path, "wt")

    meeting_count = 0
    for i, uttid in enumerate(local_all_recs):
        wav = kaldiio.load_mat(ark_scp[uttid])
        if isinstance(wav, tuple):
            if isinstance(wav[0], int):
                sr, arr = wav
            else:
                arr, sr = wav
        else:
            sr, arr = args.sample_rate, wav
        if sr != args.sample_rate:
            logging.info(f"{rank}/{threads_num} {i}/{len(local_all_recs)}({i / len(local_all_recs) * 100.0:.2f}%): skip {uttid} sr={sr}.")
        if arr.dtype == np.float32 or arr.dtype == np.float64 or np.max(np.abs(arr)) >= 2**16:
            arr = (arr / np.max(np.abs(arr)) * 0.9 * 2 ** 15).astype(np.int16)

        ark_writer(uttid, (sr, arr))
        shape_str = ",".join([str(x) for x in arr.shape])
        length_writer.write(f"{uttid} {shape_str}\n")

        if i % 100 == 0:
            logging.info(f"{rank}/{threads_num} {i}/{len(local_all_recs)}({i/len(local_all_recs)*100.0:.2f}%): process {uttid}.")
            length_writer.flush()

        meeting_count += 1

    ark_writer.close()
    length_writer.close()
    logging.info("{}/{}: Complete {} records.".format(rank, threads_num, meeting_count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ark_scp',
                        type=str,
                        default=None,
                        help="kaldi-style wav path script")
    parser.add_argument('--uttidx',
                        type=str,
                        default=None,
                        help="uttidx")
    parser.add_argument("--use_arg",
                        type=bool,
                        default=False,
                        help="weather on pai.")
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help="local rank of gpu")
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help="the world size")
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
