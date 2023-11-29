import os
import sys
import tqdm
from funcodec.text.phoneme_tokenizer import G2p_en


def main():
    in_file = sys.argv[1]
    out_file = sys.argv[2]

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    g2p = G2p_en(no_space=True)

    out_file = open(out_file, "wt", encoding="utf-8")
    content = open(in_file, "rt").readlines()
    for line in tqdm.tqdm(content, total=len(content)):
        key = os.path.basename(line.strip()).split(".", maxsplit=1)[0]
        text = open(line.strip(), "rt").readlines()[0]
        phoneme_list = g2p(text)
        # filter out other symbols
        phoneme_list = list(filter(lambda s: s != " " and s.isalnum(), phoneme_list))
        phonemes = ' '.join(phoneme_list)
        out_file.write(f"{key}\t{phonemes}\n")

    out_file.close()


if __name__ == '__main__':
    main()
