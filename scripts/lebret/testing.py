import inspect
import os
import sys

import numpy as np
import pickle
from keras.models import load_model
from odo.backends.dask import bag_to_iterator

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from config import *
from sample import Sample
from main import create_one_sample
from data_loader import load_infoboxes


def get_n_best(ls, n):
    arr = np.array(ls)
    return arr.argsort()[-n:][::-1]


def load_from_file(test_set):
    path_to_files = path + "pickle/" + dataset
    output = pickle.load(open(path_to_files + "/output.pickle", "rb"))
    # t_fields = pickle.load(open(path_to_files + "/t_fields.pickle", "rb"))
    # t_words = pickle.load(open(path_to_files + "/t_words.pickle", "rb"))
    # infoboxes = pickle.load(open(path_to_test + "/infoboxes.pickle", "rb"))
    infoboxes, u_keys = load_infoboxes(path + "data/" + test_set, test_set)
    field_transform = pickle.load(open(path_to_files + "/field_tf.pickle", "rb"))
    word_transform = pickle.load(open(path_to_files + "/word_tf.pickle", "rb"))
    model = load_model(path + "models/" + dataset + "/model_1.h5")
    return infoboxes, output, model, field_transform, word_transform


def make_sample(b):
    context, s_context, e_context = create_one_sample(b.indexes, b.starts, b.ends, -l, None, loc_dim)
    return [context], [s_context], [e_context]


def process_infoboxes(dict_list, field_transform, word_transform):
    table_fields = []
    table_words = []
    infoboxes = []
    unique_keys = set(field_transform.keys())
    field_values = set(word_transform.keys())
    for r in dict_list:
        field = {}
        table_f = set()
        table_w = set()
        for key in r:
            s = key.split("_")
            k = "_".join(s[:-1])
            idx = int(s[-1])
            word = r[key]
            if k in unique_keys:
                table_f.add(field_transform[k])

                if r[key] in field:
                    field[word].append((field_transform[k], min(l, idx), max(1, l - idx + 1)))
                else:
                    field[word] = [(field_transform[k], min(l, idx), max(1, l - idx + 1))]
            if word in field_values:
                table_w.add(word)
        table_fields.append(list(table_f))
        if len(table_w) == 0:
            table_words.append([])
        else:
            table_words.append([word_transform[w] for w in table_w])
        infoboxes.append(field)
    return table_fields, table_words, infoboxes


def beam_search(model, size, sent_length, n, output, word_tf, gf, gw, infobox):
    beam = [Sample(1.0, ["s" + str(i) for i in range(l)], [word_tf["s" + str(i)] for i in range(l - 1)], word_tf,
                   [[] for i in range(l - 1)], [[] for i in range(l - 1)], infobox)]
    # init first sample
    while True:
        new_beam = []
        for b in beam:
            if b.sentence[-1] == '.' or len(b.sentence) == sent_length:
                return b.sentence
                # predict for each element in beam
            samples_context, samples_ls, samples_le = make_sample(b)
            prediction = model.predict(
                {'c_input': np.array(samples_context), 'ls_input': np.array(samples_ls),
                 'le_input': np.array(samples_le),
                 'gf_input': np.array(gf),
                 'gw_input': np.array(gw)})
            best_pred = get_n_best(prediction[0], n)
            for p in best_pred:
                score = prediction[0][p]
                s = Sample(b.score * score * 100, b.sentence + [output[p]], b.indexes, word_tf, b.starts, b.ends, infobox)
                new_beam.append(s)
        new_score = [nb.score for nb in new_beam]
        best_scores = get_n_best(new_score, size)
        beam = [new_beam[bs] for bs in best_scores]


def test_model(model, infoboxes, f_tf, w_tf, output):
    t_fields, t_words, ib = process_infoboxes(infoboxes, f_tf, w_tf)
    i = 0
    gf = [np.pad(t_fields[i], (0, f_len - len(t_fields[i])), mode='constant')]
    gw = [np.pad(t_words[i], (0, w_len - len(t_words[i])), mode='constant')]
    print(beam_search(model, 10, 30+l, 5, output, w_tf, gf, gw, infoboxes[0]))
    pass


if __name__ == '__main__':
    infoboxes, output, model, field_transform, word_transform = load_from_file("test")
    with open(path + "samples/" + dataset + "/params.txt") as f:
        V, max_loc_idx, glob_field_dim, glob_word_dim, loc_dim, f_len, w_len, w_count = [int(a) for a in
                                                                                         f.read().split()]
    test_model(model, infoboxes, field_transform, word_transform, output)
