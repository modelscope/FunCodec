from funcodec.text.phoneme_tokenizer import G2p_en, pypinyin_g2p_phone
import sys
import os


class PhonemeTokenizer:
    """
    Style for pinyin: https://pypinyin.readthedocs.io/zh_CN/master/api.html#style. We use Style.TONE3 here
    """

    def __init__(self):
        self.g2p = G2p_en(no_space=True)

    def tokenize(self, input_file_path, output_file_path, lang=None):
        with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
            for line in input_file:
                uttid, text = line.strip().split(maxsplit=1)
                if lang == 'ch' or all(map(lambda c: '\u4e00' <= c <= '\u9fa5', ''.join(text))):
                    phoneme_list = pypinyin_g2p_phone(text)
                    phoneme_list = list(filter(lambda s: s != " ", phoneme_list))
                elif lang == 'en' or all(map(lambda c: 'a' <= c <= 'z', ''.join(text))):
                    phoneme_list = self.g2p(text)
                    # filter out other symbols
                    phoneme_list = list(filter(lambda s: s != " " and s.isalnum(), phoneme_list))
                phonemes = ' '.join(phoneme_list)
                output_file.write(f"{uttid} {phonemes}\n")

    def normalize(self, input_file_path, output_file_path):
        pass

    def generate_lexicon(self, input_file_path, output_file_path):
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
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    lang = sys.argv[3]
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    g2p_tokenizer = PhonemeTokenizer()
    g2p_tokenizer.tokenize(in_file, out_file, lang)
