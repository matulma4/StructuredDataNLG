# -*- coding: utf-8 -*-
import pickle
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from config import *


def get_occurring(ls, n):
	"""
	Obtain elements of list occuring at least n times in it

    Keyword arguments:
    ls -- a list
    n -- minimum number of times the element needs to appear in the list (int)
    """
    counts = Counter(ls)
    return [i for i in counts if counts[i] >= n]


def load_infoboxes(path, dataset):
	"""
	Load tables from a file

    Keyword arguments:
    path -- path to file
    dataset -- which dataset should be loaded
    """
    result = []
    i = 0
    unique_keys = []
    with open(path + "/" + dataset + ".box", encoding="utf-8") as f:
        for person_box in f:
            info = [k.split(":") for k in person_box.strip().split("\t")]
			# Use only pairs where the value is not <none>
            result.append(dict([(v[0], v[1]) for v in info if v[1] != "<none>"]))
            unique_keys += ["_".join(s.split("_")[:-1]) for s in result[-1].keys()]
            i += 1
            if i == limit:
                break
    unique_keys = get_occurring(unique_keys, min_keys)
    return result, set(unique_keys)


def load_sentences(pth, d_set):
	"""
	Load natural language sentences from a file

    Keyword arguments:
    pth -- path to file
    d_set -- which dataset should be loaded
    """
    punc = [",", ".", "-lrb-","-rrb-",'``',"''"] if drop_punc else []
    counts = [int(n.strip()) for n in open(pth + "/" + d_set + ".nb")][:limit]
    result = []
    with open(pth + "/" + d_set + ".sent", encoding="utf-8") as f:
        for person_index in range(len(counts)):
            sent_count = 0
            person = []
            while sent_count < counts[person_index]:
                person.append(f.readline().strip())

                sent_count += 1
            result.append([p for p in person[0].split() if p not in punc])
    return result



def create_vocabulary(sents):
	"""
	Create a vocabulary from a list of sentences

    Keyword arguments:
    sents -- list of sentences
    """
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sents)
    y = np.sum(X, axis=0)
    names = vectorizer.get_feature_names()
    cnts = [x for _, x in sorted(zip(np.array(y)[0, :], names), reverse=True)]
    return dict([(b, a) for a, b in enumerate(cnts)])

