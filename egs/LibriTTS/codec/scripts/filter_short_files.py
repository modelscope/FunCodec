import kaldiio
import sys
import os
from funcodec.utils.misc import get_logger
import logging


if __name__ == '__main__':
    logger = get_logger(log_level=logging.INFO)
    in_scp = sys.argv[1]
    out_scp = sys.argv[2]

    if not os.path.exists(os.path.dirname(out_scp)):
        os.makedirs(os.path.dirname(out_scp))

    out_fd = open(out_scp, "wt", encoding="utf-8")
    count = 0
    for one_line in open(in_scp, "rt"):
        uttid, wav_path = one_line.strip().split(" ", maxsplit=1)
        wav_path = wav_path.replace("/data/volume1/", "/nfs/")
        wav = kaldiio.load_mat(wav_path)
        if len(wav) < 0.5*16000:
            continue
        out_fd.write(f"{uttid} {wav_path}\n")
        if count % 1000 == 0:
            logger.info(f"completed {count}")
        count += 1

    out_fd.close()
