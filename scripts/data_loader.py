# -*- coding: utf-8 -*-
import os
import pickle
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

limit = 5000


def transform_to_indices(data):
    values = np.array([b for a in data for b in a])
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    i = 0
    result = []
    for d in data:
        result.append(integer_encoded[i:i + len(d)])
        i += len(d)
    return result


def load_infoboxes():
    result = []
    fields = []
    with open(path + "/" + dataset + ".box", encoding="utf-8") as f:
        for person_box in f:
            info = [k.split(":") for k in person_box.strip().split("\t")]
            # result.append(dict([(v[0], v[1]) for v in info if v[1] != "<none>"]))
            d = list(zip(*[(v[0], v[1]) for v in info if v[1] != "<none>"]))
            fields.append(list(d[0]))
            result.append(list(d[1]))

    # fields = [list(r.keys()) for r in result]
    return result[:limit], fields[:limit]


def get_keys(fields):
    if not os.path.exists(path + "/" + dataset + ".counts.pickle"):
        counts = {}
        for d in fields:
            for b in d:
                # c = "_".join(b.split("_")[:-1])
                c = b
                if c in counts.keys():
                    counts[c] += 1
                else:
                    counts[c] = 1
        pickle.dump(counts, open(path + "/" + dataset + ".counts.pickle", "wb"))
    else:
        counts = pickle.load(open(path + "/" + dataset + ".counts.pickle", "rb"))
    known = []
    for key in counts.keys():
        if counts[key] > 100:
            known.append(key)
    return known


def load_sentences():
    counts = [int(n.strip()) for n in open(path + "/" + dataset + ".nb")]#[:limit]
    result = []
    all_sents = []
    with open(path + "/" + dataset + ".sent", encoding="utf-8") as f:
        for person_index in range(len(counts)):
            sent_count = 0
            person = []
            while sent_count < counts[person_index]:
                person.append(re.sub("[0-9]+","<NUMBER>", re.sub("([1-2]?[0-9]{3}|3000)", "<YEAR>", f.readline().strip())))
                # all_sents.append(person[-1])
                sent_count += 1
            result.append(person[0].split())
    return result# , all_sents


def create_vocabulary(sents):
    # t = [a for s in sents for a in s.split()]
    # return list(set(t))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sents)
    y = np.sum(X, axis=0)
    names = vectorizer.get_feature_names()
    cnts = [x for _, x in sorted(zip(np.array(y)[0, :], names), reverse=True)]
    return dict([(b, a) for a, b in enumerate(cnts)])


def get_most_frequent(all_sents):
    words = [a for b in all_sents for a in b]
    c = Counter(words)
    result = [a[0] for a in c.most_common(20000)]
    return result


if __name__ == '__main__':
    dataset = "valid"
    if "nt" == os.name:
        path = "E:/Martin/PyCharm Projects/StructuredDataNLG/data/" + dataset
    else:
        path = "/data/matulma4/wikipedia-biography-dataset/wikipedia-biography-dataset/" + dataset
    sentences = load_sentences()
    get_most_frequent(sentences)

    r, f = load_infoboxes()
    transform_to_indices(r)
    if not os.path.exists(path + "/" + dataset + ".key"):
        k = sorted(get_keys(f))
        with open(path + "/" + dataset + ".key", "w") as f:
            for key in k:
                f.write(key + "\n")
    # voc = create_vocabulary(sentences)
    pass
