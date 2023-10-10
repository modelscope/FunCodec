import os
import sys
import numpy as np
import kaldiio
from sklearn import cluster
import logging
import random

logging.basicConfig(
    level='INFO',
    format=f"[{os.uname()[1].split('.')[0]}]"
           f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)

if __name__ == '__main__':
    feat_file = sys.argv[1]
    num_cluster = int(sys.argv[2])
    out_dir = sys.argv[3]
    use_batch = True

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logging.info("loading data...")
    key_list, mat_list = [], []
    for line in open(feat_file, "rt"):
        key, path = line.strip().split(" ", maxsplit=1)
        key_list.append(key)
        arr = kaldiio.load_mat(path)
        tt = arr.shape[0]
        arr = arr.reshape((tt, 8, -1))
        arr = np.sum(arr, axis=1, keepdims=False)
        mat_list.append(arr)

    random.shuffle(mat_list)
    logging.info("concatenating data...")
    X = np.concatenate(mat_list, axis=0)
    logging.info(f"Total {X.shape[0]} samples.")
    np.random.shuffle(X)
    logging.info("clustering data...")
    if use_batch:
        logging.info("use mini-batch kmeans")
        kmeans = cluster.MiniBatchKMeans(
            n_clusters=num_cluster,
            init="random",
            batch_size=8192*16,
            max_iter=20,
            n_init="auto",
            compute_labels=True,
            verbose=True,
            tol=0.0,
            max_no_improvement=None,
        ).fit(X)
    else:
        kmeans = cluster.KMeans(
            n_clusters=num_cluster,
            init="random",
            n_init="auto",
            max_iter=100,
            verbose=True,
        ).fit(X)

    logging.info("saving data...")
    cluster_centers, labels = kmeans.cluster_centers_, kmeans.labels_
    np.save(os.path.join(out_dir, "centroid.npy"), cluster_centers)
    np.save(os.path.join(out_dir, "labels.npy"), labels)

    # !! The mat_list has been shuffled. can not be used !!
    # idx_file = open(os.path.join(out_dir, "idx.txt"), "wt")
    # outfile_path = os.path.join(out_dir, "feats")
    # ark_writer = kaldiio.WriteHelper("ark,scp,f:{}.ark,{}.scp".format(outfile_path, outfile_path))
    #
    # for key, mat in zip(key_list, mat_list):
    #     lbl = kmeans.predict(mat)
    #     str_lbl = [str(x) for x in lbl.tolist()]
    #     str_lbl = " ".join(str_lbl)
    #     idx_file.write(f"{key} {str_lbl}\n")
    #
    #     x = [cluster_centers[i] for i in lbl]
    #     x = np.row_stack(x)
    #     ark_writer(key, x)
    #
    # idx_file.close()
    # ark_writer.close()
