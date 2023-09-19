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
import numpy as np
from funcodec.utils.misc import get_logger
import kaldiio
from sklearn import cluster


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

    centroids = np.load(args.centroid)
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=centroids.shape[0],
        init="random",
        batch_size=8192,
        max_iter=1,
        n_init="auto",
        compute_labels=True,
        verbose=False,
        tol=0.0,
        max_no_improvement=None,
    ).fit(np.concatenate([centroids, centroids], axis=0))
    kmeans.cluster_centers_ = centroids

    idx_file = open(os.path.join(out_dir, f"idx.{rank}.txt"), "wt")
    outfile_path = os.path.join(out_dir, f"feats.{rank}")
    ark_writer = kaldiio.WriteHelper("ark,scp,f:{}.ark,{}.scp".format(outfile_path, outfile_path))

    process_count = 0
    for i, (uttid, mat_path) in enumerate(local_all_recs):
        arr = kaldiio.load_mat(mat_path)
        tt = arr.shape[0]
        arr = arr.reshape((tt, 8, -1))
        arr = np.sum(arr, axis=1, keepdims=False)
        labels = kmeans.predict(arr)
        str_labels = " ".join(labels.astype(str).tolist())
        idx_file.write(f"{uttid} {str_labels}\n")
        idx_file.flush()
        x = [centroids[j] for j in labels]
        x = np.row_stack(x)
        ark_writer(uttid, x)

        if i % 100 == 0:
            logger.info("{}/{}: process {}.".format(rank, threads_num, uttid))

        process_count += 1

    idx_file.close()
    ark_writer.close()
    logger.info("{}/{}: Complete {} records.".format(rank, threads_num, process_count))


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
    parser.add_argument('--centroid',
                        type=str,
                        default=None,
                        help="The npy file contains centroids.")
    args = parser.parse_args()
    main(args)
