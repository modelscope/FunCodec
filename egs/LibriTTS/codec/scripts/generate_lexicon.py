import os
import sys


def generate_lexicon(input_file_path, output_file_path):
    lexicon = dict()
    additional_keys = ['<s>', '</s>', '<unk>', '<pad>']  # edit here
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            uttid, *sentence = line.strip().split()
            for word in sentence:
                if word not in lexicon:
                    lexicon[word] = 1
                else:
                    lexicon[word] += 1
        for key in additional_keys:
            lexicon[key] = 0
        for key, value in sorted(lexicon.items(), key=lambda item: item[1], reverse=True):
            output_file.write(f"{key} {value}\n")


if __name__ == '__main__':
    trans_file = sys.argv[1]
    lexicon_file = sys.argv[2]

    os.makedirs(os.path.dirname(lexicon_file), exist_ok=True)

    generate_lexicon(trans_file, lexicon_file)
