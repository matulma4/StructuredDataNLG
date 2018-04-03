# -*- coding: utf-8 -*-
import pickle
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from config import *


def get_occurring(ls, n):
    counts = Counter(ls)
    return [i for i in counts if counts[i] >= n]


def load_infoboxes(path, dataset):
    result = []
    i = 0
    unique_keys = []
    with open(path + "/" + dataset + ".box", encoding="utf-8") as f:
        for person_box in f:
            info = [k.split(":") for k in person_box.strip().split("\t")]
            result.append(dict([(v[0], v[1]) for v in info if v[1] != "<none>"]))
            unique_keys += ["_".join(s.split("_")[:-1]) for s in result[-1].keys()]
            # TODO remove limiter
            i += 1
            if i == limit:
                break
    unique_keys = get_occurring(unique_keys, 10)
    return result, set(unique_keys)


def load_sentences():
    counts = [int(n.strip()) for n in open(data_path + "/" + dataset + ".nb")][:limit]
    result = []
    with open(data_path + "/" + dataset + ".sent", encoding="utf-8") as f:
        for person_index in range(len(counts)):
            sent_count = 0
            person = []
            while sent_count < counts[person_index]:
                # person.append(re.sub("[0-9]+","<NUMBER>", re.sub("([1-2]?[0-9]{3}|3000)", "<YEAR>", f.readline().strip())))
                person.append(f.readline().strip())

                sent_count += 1
            result.append(person[0].split())
    return result


# TODO remove?
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


def create_vocabulary(sents):
    # t = [a for s in sents for a in s.split()]
    # return list(set(t))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sents)
    y = np.sum(X, axis=0)
    names = vectorizer.get_feature_names()
    cnts = [x for _, x in sorted(zip(np.array(y)[0, :], names), reverse=True)]
    return dict([(b, a) for a, b in enumerate(cnts)])


if __name__ == '__main__':
    pass
