# -*- coding: utf-8 -*-
import os

import pickle


def load_infoboxes():
    result = []
    with open(path + "/" + dataset + ".box", encoding="utf-8") as f:
        for person_box in f:
            info = [k.split(":") for k in person_box.strip().split("\t")]
            result.append(dict([(v[0], v[1]) for v in info if v[1] != "<none>"]))
    fields = [list(r.keys()) for r in result]
    return result, fields


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
    counts = [int(n.strip()) for n in open(path + "/" + dataset + ".nb")]
    result = []
    with open(path + "/" + dataset + ".sent", encoding="utf-8") as f:
        for person_index in range(len(counts)):
            sent_count = 0
            person = []
            while sent_count < counts[person_index]:
                person.append(f.readline().strip())
                sent_count += 1
            result.append(person)
    return result


if __name__ == '__main__':
    dataset = "valid"
    if "nt" == os.name:
        path = "E:/Martin/PyCharm Projects/StructuredDataNLG/data/" + dataset
    else:
        path = "/data/matulma4/wikipedia-biography-dataset/wikipedia-biography-dataset/" + dataset
    r, f = load_infoboxes()
    if not os.path.exists(path + "/" + dataset + ".key"):
        k = sorted(get_keys(f))
        with open(path + "/" + dataset + ".key", "w") as f:
            for key in k:
                f.write(key + "\n")
    load_sentences()
