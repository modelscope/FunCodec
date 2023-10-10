import json
import os
import sys
from tqdm import tqdm


if __name__ == '__main__':
    in_file = sys.argv[1]
    out_file = sys.argv[2]

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    out_file = open(out_file, "wt")
    count = 0
    for line in open(in_file, "rt"):
        uttid, data_json = line.strip().split(maxsplit=1)
        data_dict: dict = json.loads(data_json)
        input_data = data_dict["input"]
        wav_len = len(input_data) * 640
        out_file.write(f"{uttid.replace('_ASR', '')} {wav_len}\n")

        if (count-1) % 10000 == 0:
            out_file.flush()
            print(f"Processed {count} utterances.")

        count += 1

    out_file.close()
    print(f"Total {count} utterances.")
