# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import pypinyin
from pypinyin import pinyin
from zhconv import convert


class TrieNode(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.data = {}
        self.is_word = False


class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        node = self.root
        for chars in word:
            child = node.data.get(chars)
            if not child:
                node.data[chars] = TrieNode()
            node = node.data[chars]
        node.is_word = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node = self.root
        for chars in word:
            node = node.data.get(chars)
            if not node:
                return False
        return node.is_word

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        node = self.root
        for chars in prefix:
            node = node.data.get(chars)
            if not node:
                return False
        return True

    def get_start(self, prefix):
        """
          Returns words started with prefix
          :param prefix:
          :return: words (list)
        """

        def get_key(pre, pre_node):
            word_list = []
            if pre_node.is_word:
                word_list.append(pre)
            for x in pre_node.data.keys():
                word_list.extend(get_key(pre + str(x), pre_node.data.get(x)))
            return word_list

        words = []
        if not self.startsWith(prefix):
            return words
        if self.search(prefix):
            words.append(prefix)
            return words
        node = self.root
        for chars in prefix:
            node = node.data.get(chars)
        return get_key(prefix, node)


class TrieTokenizer(Trie):

    def __init__(self, dict_path):
        super(TrieTokenizer, self).__init__()
        self.dict_path = dict_path
        self.create_trie_tree()
        self.punctuations = """！？｡＂＃＄％＆＇：（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."""

    def load_dict(self):
        words = []
        with open(self.dict_path, mode="r", encoding="utf-8") as file:
            for line in file:
                words.append(line.strip().split("\t")[0].encode('utf-8').decode('utf-8-sig'))
        print(words)
        return words

    def create_trie_tree(self):
        words = self.load_dict()
        for word in words:
            self.insert(word)

    def mine_tree(self, tree, sentence, trace_index):
        if trace_index <= (len(sentence) - 1):
            if sentence[trace_index] in tree.data:
                trace_index = trace_index + 1
                trace_index = self.mine_tree(tree.data[sentence[trace_index - 1]], sentence, trace_index)
        return trace_index

    def tokenize(self, sentence):
        tokens = []
        sentence_len = len(sentence)
        while sentence_len != 0:
            trace_index = 0
            trace_index = self.mine_tree(self.root, sentence, trace_index)

            if trace_index == 0:
                # print(sentence[0:1])
                tokens.append(sentence[0:1])
                sentence = sentence[1:len(sentence)]
                sentence_len = len(sentence)
            else:
                tokens.append(sentence[0:trace_index])
                sentence = sentence[trace_index:len(sentence)]  #
                sentence_len = len(sentence)

        return tokens

    def combine(self, token_list):
        flag = 0
        output = []
        temp = []
        for i in token_list:
            if len(i) != 1:
                if flag == 0:
                    output.append(i[::])
                else:
                    output.append("".join(temp))
                    output.append(i[::])
                    temp = []
                    flag = 0
            else:
                if flag == 0:
                    temp.append(i)
                    flag = 1
                else:
                    temp.append(i)
        return output


class NewText2PhoneLong:
    def __init__(self, phone_dict_path, use_alignment=False):
        self.trie_cws = TrieTokenizer(phone_dict_path)
        self.phone_map, self.alignment_map = self.get_phone_map(phone_dict_path)
        self.use_alignment = use_alignment

    def get_phone_map(self, phone_dict_path):
        phone_map_file_reader = open(phone_dict_path, "r")
        phone_map = dict()
        alignment_map = dict()
        for line in phone_map_file_reader:
            key, phone_series, alignments = line.strip().split("\t")
            if key not in phone_map:
                phone_map[key] = phone_series
                alignment_map[key] = alignments
        return phone_map, alignment_map

    def normalize(self, text):
        chinese_number = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
        new_text = ""
        for x in text:
            if x in "0123456789":
                x = chinese_number[0]
            new_text += x
        new_text = convert(new_text, 'zh-hans')
        return new_text

    def trans(self, text):
        if text == "<unk>":
            if self.use_alignment:
                return "<unk>", [1]
            else:
                return "<unk>"
        text = self.normalize(text)
        tokens = self.trie_cws.tokenize(text)
        # print(tokens)
        phones = []
        alignments = []
        for word in tokens:
            if word in self.phone_map:
                phones.append(self.phone_map[word])
                alignments.append(self.alignment_map[word])
            elif len(word) > 1:
                for char in word:
                    if char in self.phone_map:
                        # print("char: {}".format(char))
                        phones.append(self.phone_map[char])
                        alignments.append(self.alignment_map[word])
                    else:
                        phone = pinyin(char)[0][0]
                        phone1 = pypinyin.contrib.tone_convert.to_initials(phone, strict=False)
                        phone2 = pypinyin.contrib.tone_convert.to_finals_tone2(phone, strict=False)
                        flag = False
                        for num in ["1", "2", "3", "4"]:
                            if num in phone2 and phone2[-1] != num:
                                flag = True
                                phone2_parts = phone2.split(num)
                                phone2 = "{}{} {}".format(phone2_parts[0], num, phone2_parts[1])
                        if phone1 == "":
                            if flag:
                                alignments.append("2")
                            else:
                                alignments.append("1")
                            phones.append(phone2)
                        else:
                            if flag:
                                alignments.append("3")
                            else:
                                alignments.append("2")
                            phones.append("{} {}".format(phone1, phone2))
            else:
                phone = pinyin(word)[0][0]
                phone1 = pypinyin.contrib.tone_convert.to_initials(phone, strict=False)
                phone2 = pypinyin.contrib.tone_convert.to_finals_tone2(phone, strict=False)
                flag = False
                for num in ["1", "2", "3", "4"]:
                    if num in phone2 and phone2[-1] != num:
                        flag = True
                        phone2_parts = phone2.split(num)
                        phone2 = "{}{} {}".format(phone2_parts[0], num, phone2_parts[1])
                if phone1 == "":
                    if flag:
                        alignments.append("2")
                    else:
                        alignments.append("1")
                    phones.append(phone2)
                else:
                    if flag:
                        alignments.append("3")
                    else:
                        alignments.append("2")
                    phones.append("{} {}".format(phone1, phone2))
        if self.use_alignment:
            alignments = [int(ali) for ali in " ".join(alignments).split()]
            return " ".join(phones), alignments
        else:
            return " ".join(phones)

    def double_trans(self, text):
        if text == "<unk>":
            if self.use_alignment:
                return "<unk> <unk>", [2]
            else:
                return "<unk> <unk>"
        text = self.normalize(text)
        tokens = self.trie_cws.tokenize(text)
        # print(tokens)
        phones = []
        alignments = []
        for word in tokens:
            if word in self.phone_map:
                phones.append(self.phone_map[word])
                alignments.append(self.alignment_map[word])
            elif len(word) > 1:
                for char in word:
                    if char in self.phone_map:
                        # print("char: {}".format(char))
                        phones.append(self.phone_map[char])
                        alignments.append(self.alignment_map[word])
                    else:
                        phone = pinyin(char)[0][0]
                        phone1 = pypinyin.contrib.tone_convert.to_initials(phone, strict=False)
                        phone2 = pypinyin.contrib.tone_convert.to_finals_tone2(phone, strict=False)
                        flag = False
                        for num in ["1", "2", "3", "4"]:
                            if num in phone2 and phone2[-1] != num:
                                flag = True
                                phone2_parts = phone2.split(num)
                                phone2 = "{}{} {}".format(phone2_parts[0], num, phone2_parts[1])
                        if phone1 == "":
                            if flag:
                                alignments.append("2")
                            else:
                                alignments.append("1")
                            phones.append(phone2)
                        else:
                            if flag:
                                alignments.append("3")
                            else:
                                alignments.append("2")
                            phones.append("{} {}".format(phone1, phone2))
            else:
                phone = pinyin(word)[0][0]
                phone1 = pypinyin.contrib.tone_convert.to_initials(phone, strict=False)
                phone2 = pypinyin.contrib.tone_convert.to_finals_tone2(phone, strict=False)
                flag = False
                for num in ["1", "2", "3", "4"]:
                    if num in phone2 and phone2[-1] != num:
                        flag = True
                        phone2_parts = phone2.split(num)
                        phone2 = "{}{} {}".format(phone2_parts[0], num, phone2_parts[1])
                if phone1 == "":
                    if flag:
                        alignments.append("2")
                    else:
                        alignments.append("1")
                    phones.append(phone2)
                else:
                    if flag:
                        alignments.append("3")
                    else:
                        alignments.append("2")
                    phones.append("{} {}".format(phone1, phone2))

        # double phones and alignments
        phones = " ".join(phones).split()
        phones_double = [" ".join([p, p]) for p in phones]
        phones_double = " ".join(phones_double)

        alignments = " ".join(alignments).split()
        alignments_double = [str(int(ali) * 2) for ali in alignments]
        alignments_double = [int(ali) for ali in " ".join(alignments_double).split()]

        if self.use_alignment:
            return phones_double, alignments_double
        else:
            return phones_double


if __name__ == '__main__':
    text2phone_tokenizer = NewText2PhoneLong("text2phone_from_pinyin_dict_long.txt", use_alignment=True)
    # sentence = "讧长长久久"
    sentence = "我是谁长久最大楼市饿内讧红讧恶心"
    phone_seq, alignment_seq = text2phone_tokenizer.double_trans(sentence)
    print(phone_seq)
    print(alignment_seq)
