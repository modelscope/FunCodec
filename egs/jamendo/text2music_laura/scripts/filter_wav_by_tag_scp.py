import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_scp", type=str)
    parser.add_argument("--tag_scp", type=str)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tag_dict = {}
    tag_scp_fd = open(args.tag_scp, "rt")
    for line in tag_scp_fd:
        reco_id, tags = line.strip().split("\t", maxsplit=1)
        tag_dict[reco_id] = tags

    out_wav_scp_fd = open(os.path.join(args.out_dir, "wav.scp"), "wt")
    out_text_fd = open(os.path.join(args.out_dir, "text"), "wt")
    wav_scp_fd = open(args.wav_scp, "rt")
    for line in wav_scp_fd:
        uttid, path = line.strip().split(maxsplit=1)
        reco_id = uttid.rsplit("-", maxsplit=1)[0]
        if reco_id in tag_dict:
            out_wav_scp_fd.write(f"{uttid}\t{path}\n")
            out_text_fd.write(f"{uttid}\t{tag_dict[reco_id]}\n")

    out_wav_scp_fd.close()
    out_text_fd.close()


if __name__ == '__main__':
    main()
