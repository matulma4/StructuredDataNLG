import re
from collections import Counter

import numpy as np
from sklearn.preprocessing import LabelEncoder

from config import *
from data_loader import load_sentences, load_infoboxes


def get_words(sentences):
    vocabulary = get_most_frequent(sentences, True)
    sents = replace_oov(sentences, vocabulary)
    indices, encoder = transform_to_indices(sents)
    return indices, encoder


def local_conditioning(tf, f, sentences):
    F = len(tf)
    start = []
    end = []
    max_len = 0
    for i in range(len(f)):
        ib = f[i]
        sent = sentences[i]
        s_seq = []
        e_seq = []
        for word in sent:
            if word in ib.keys():
                s_seq.append(list(set([j[1] + j[0] * l for j in ib[word]])))
                e_seq.append(list(set([j[2] + j[0] * l for j in ib[word]])))
                max_len = max(len(s_seq[-1]), max_len)
                max_len = max(len(e_seq[-1]), max_len)
            else:
                s_seq.append([F * l + 1])
                e_seq.append([F * l + 1])
        start.append(s_seq)
        end.append(e_seq)
    print("Maximum occurrences of words: " + str(max_len))
    return np.array(start), np.array(end)


def replace_oov(sentences, vocabulary):
    for i in range(len(sentences)):
        sentence = sentences[i]
        for j in range(len(sentence)):
            word = sentence[j]
            if word not in vocabulary:
                sentence[j] = "<UNK>"
        sentences[i] = ["s" + str(i) for i in range(l)] + sentence
    return sentences


def transform_to_indices(data):
    values = np.array([b for a in data for b in a])
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    i = 0
    result = []
    for d in data:
        result.append(integer_encoded[i:i + len(d)])
        i += len(d)
    return result, integer_encoded


def get_most_frequent(all_sents, sub_numbers):
    if sub_numbers:
        words = [re.sub("[0-9]+", "<NUMBER>", re.sub("([1-2]?[0-9]{3}|3000)", "<YEAR>", a)) for b in all_sents for a in
                 b]
    else:
        words = [re.sub("[0-9]{2}", "10", a) for b in all_sents for a in b]
    c = Counter(words)
    result = [a[0] for a in c.most_common(20002) if a[0] != "<NUMBER>" and a[0] != "<YEAR>"]
    return result


def process_infoboxes(unique_keys, dict_list):
    fields = []
    le = LabelEncoder()
    tf = dict(zip(unique_keys, le.fit_transform(unique_keys)))
    for r in dict_list:
        field = {}
        for key in r:
            s = key.split("_")
            k = "_".join(s[:-1])
            idx = int(s[-1])
            if k in unique_keys:
                if r[key] in field:
                    field[r[key]].append((tf[k], min(l, idx), max(1, l - idx + 1)))
                else:
                    field[r[key]] = [(tf[k], min(l, idx), max(1, l - idx + 1))]
        fields.append(field)
    return fields, tf


def delexicalize(sentences, tables, vocabulary):
    for i in range(len(sentences)):
        table = tables[i]
        sentence = sentences[i]
        for j in range(len(sentence)):
            word = sentence[j]
            if word not in vocabulary:
                for k in table.keys():
                    if table[k] == word:
                        sentence[j] = re.sub("[0-9]{2}", "10", k)
                        break
    return sentences


if __name__ == '__main__':
    dicts, u_keys = load_infoboxes(path, dataset)
    f, tf = process_infoboxes(u_keys, dicts)
    sentences = load_sentences()
    indices, encoder = get_words(sentences)
    print("Size of vocabulary: " + str(max(encoder)))
    print("Number of fields: " + str(len(tf)))
    local_conditioning(tf, f, sentences)
    sentences = delexicalize(sentences, dicts, encoder.classes_)

    glob_f_vec = np.arange(len(tf))
    glob_w_vec = np.arange(max(encoder))
