# -*- coding: utf-8 -*-
import os


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

    load_sentences()
