import csv
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv_file",
        type=str,
    )
    parser.add_argument(
        "--out_file",
        type=str,
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    out_fd = open(args.out_file, "wt")
    with open(args.tsv_file) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for i, row in enumerate(rd):
            if i == 0:
                continue
            mp3_path = row[3]
            rec_id = mp3_path.replace("/", "-").replace(".mp3", "")
            tag_list = row[5:]
            disc_dict = {}
            for tag in tag_list:
                tag_name, tag_value = tag.split("---", maxsplit=1)
                if tag_name not in disc_dict:
                    disc_dict[tag_name] = []
                disc_dict[tag_name].append(tag_value)
            tag_str = []
            for key in ["genre", "instrument", "mood/theme"]:
                if key in disc_dict:
                    tag_str.append("{}: {}".format(key, ", ".join(disc_dict[key])))
            tag_str = "; ".join(tag_str)

            out_fd.write(f"{rec_id}\t{tag_str}\n")

    out_fd.close()


if __name__ == '__main__':
    main()
