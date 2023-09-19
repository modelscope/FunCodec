import os
import kaldiio
from kaldiio import ReadHelper
import sys
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    wav_scp = sys.argv[1]
    length_file = sys.argv[2]

    os.makedirs(os.path.dirname(length_file), exist_ok=True)
    num_lines = len(open(wav_scp, "rt").readlines())

    length_file = open(length_file, "wt", encoding="utf-8")
    for key, retval in tqdm(ReadHelper(f"scp:{wav_scp}"), total=num_lines, ascii=True):
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

        length_file.write(f"{key} {array.shape[0]}\n")

    length_file.close()
