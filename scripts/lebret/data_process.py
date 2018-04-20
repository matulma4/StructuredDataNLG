import inspect
import os
import re
import sys
import pickle
import time
from collections import Counter
import fastText.FastText as ft
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
from sklearn.preprocessing import LabelEncoder

from config import *
from data_loader import load_sentences, load_infoboxes


def get_words(sentences):
    vocabulary = get_most_frequent(sentences, True)
    if use_ft:
        ft_model = ft.load_model(data_path + "/" + dataset + "_vectors.bin")
        ft_vectors = np.array([ft_model.get_word_vector(v) for v in vocabulary])
    else:
        ft_vectors = None
    # if False:#os.path.exists(data_path + "/" + dataset + ".oov"):
    #     sents = [line.strip().split(" ") for line in open(data_path + "/" + dataset + ".oov", encoding='utf-8')]
    # else:
    sents = replace_oov(sentences, set(vocabulary))
        # with open(data_path + "/" + dataset + ".oov", "w", encoding='utf-8') as g:
        #     for sent in sents:
        #         g.write(" ".join(sent) + "\n")
    indices, encoder, max_word_idx = transform_to_indices(sents)
    return indices, encoder, max_word_idx, ft_vectors


def local_conditioning(tf, f, sentences):
    F = len(tf)
    start = []
    end = []
    max_len = 0
    # TODO fix indices from end
    for i in range(len(f)):
        ib = f[i]
        sent = sentences[i]
        s_seq = [[F * l + 1] for _ in range(l)]
        e_seq = [[F * l + 1] for _ in range(l)]
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
    return np.array(start), np.array(end), max_len


def replace_oov(sentences, vocabulary):
    new_sentences = []
    for i in range(len(sentences)):
        new_sentence = []
        sentence = sentences[i]
        for j in range(len(sentence)):
            word = sentence[j]
            if word not in vocabulary:
                new_sentence.append("<UNK>")
            else:
                new_sentence.append(sentence[j])
        new_sentences.append(["s" + str(i) for i in range(l)] + new_sentence)
    return new_sentences


def transform_to_indices(data):
    values = list(set(([b for a in data for b in a])))
    label_encoder = LabelEncoder()
    label_encoder.fit(values)
    i = 0
    result = [label_encoder.transform(d) for d in data]
    # for d in data:
    #     result.append(integer_encoded[i:i + len(d)])
    #     i += len(d)
    return result, label_encoder, max(label_encoder.transform(values))


def get_most_frequent(all_sents, sub_numbers):
    if sub_numbers:
        words = [re.sub("[0-9]+", "<NUMBER>", re.sub("([1-2]?[0-9]{3}|3000)", "<YEAR>", a)) for b in all_sents for a in
                 b]
    else:
        words = [re.sub("[0-9]{2}", "10", a) for b in all_sents for a in b]
    c = Counter(words)
    result = [a[0] for a in c.most_common(V + 2) if a[0] != "<NUMBER>" and a[0] != "<YEAR>"]
    return result


def process_infoboxes(unique_keys, dict_list, encoder):
    global f_size, w_size, f_len, w_len, w_count
    infoboxes = []
    u_k = list(unique_keys)
    le = LabelEncoder()
    field_transform = dict(zip(u_k, le.fit_transform(u_k)))
    table_fields = []
    table_words = []
    field_values = set(encoder.classes_)
    f_len = 0
    w_len = 0
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
            table_words.append(encoder.transform(list(table_w)))
            w_size = max(w_size, max(table_words[-1]))

        if len(table_f) == 0:
            table_fields.append([])
        else:
            table_fields.append(list(table_f))
            # w_size = max(w_size, max(table_words[-1]))
            f_size = max(f_size, max(table_fields[-1]))
        f_len = max(f_len, len(table_fields[-1]))
        w_len = max(w_len, len(table_words[-1]))
        w_count = max(w_count, len(field))
        infoboxes.append(field)
    print("Maximum fields in table: " + str(f_len))
    print("Maximum words in table: " + str(w_len))
    return infoboxes, field_transform, table_fields, table_words, field_values


def delexicalize(sentences, tables, vocabulary, keys):
    field_names = set()
    g = open(data_path + "/" + dataset + ".dlx", "w", encoding="utf-8")
    for i in range(len(sentences)):
        table = tables[i]
        sentence = sentences[i]
        for j in range(len(sentence)):
            word = sentence[j]
            if word not in vocabulary:
                for k in table.keys():
                    if table[k] == word:
                        sentence[j] = re.sub("[0-9]{2}", str(l), k)
                        field_name = "_".join(k.split("_")[:-1])
                        if field_name in keys:
                            field_names.add(k)
                        else:
                            field_names.add("<UNK>")
                        break
        g.write(" ".join(sentence) + "\n")
    g.close()
    return list(field_names)


def save_to_file(output, indices, start, end, t_fields, t_words, infoboxes, field_transform, word_transform, vectors):
    path_to_files = path + "pickle/" + dataset
    pickle.dump(output, open(path_to_files + "/output.pickle", "wb"))
    pickle.dump(start, open(path_to_files + "/start.pickle", "wb"))
    pickle.dump(end, open(path_to_files + "/end.pickle", "wb"))
    pickle.dump(indices, open(path_to_files + "/indices.pickle", "wb"))
    pickle.dump(t_fields, open(path_to_files + "/t_fields.pickle", "wb"))
    pickle.dump(t_words, open(path_to_files + "/t_words.pickle", "wb"))
    pickle.dump(infoboxes, open(path_to_files + "/infoboxes.pickle", "wb"))
    pickle.dump(field_transform, open(path_to_files + "/field_tf.pickle", "wb"))
    pickle.dump(word_transform, open(path_to_files + "/word_tf.pickle", "wb"))
    pickle.dump(vectors, open(path_to_files + "/vectors.pickle", "wb"))


if __name__ == '__main__':
    # strt = time.time()
    dicts, u_keys = load_infoboxes(data_path, dataset)
    sentences = load_sentences()
    # print("Loading: " + str(time.time() - strt))
    # strt = time.time()
    f_size = 0
    w_size = 0
    f_len = 0
    w_len = 0
    w_count = 0
    indices, encoder, max_word_idx, vectors = get_words(sentences)
    word_transform = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    # print("get_words: " + str(time.time() - strt))
    # strt = time.time()
    infoboxes, field_transform, t_fields, t_words, field_values = process_infoboxes(u_keys, dicts, encoder)

    # print("process_infoboxes: " + str(time.time() - strt))
    # strt = time.time()
    print("Size of vocabulary: " + str(max_word_idx))
    start, end, loc_dim = local_conditioning(field_transform, infoboxes, sentences)
    # print("local_conditioning: " + str(time.time() - strt))
    # strt = time.time()
    print("Number of fields: " + str(loc_dim))
    f_names = delexicalize(sentences, dicts, field_values, u_keys)
    output = np.concatenate((encoder.classes_, f_names))
    # print("delex: " + str(time.time() - strt))
    # strt = time.time()
    save_to_file(output, indices, start, end, t_fields, t_words, infoboxes, field_transform, word_transform, vectors)
    with open(path + "pickle/" + dataset + "/params.txt", "w") as g:
        g.write(" ".join(
            [str(max_word_idx), str(len(field_transform) * l + 2), str(f_size), str(w_size), str(loc_dim), str(f_len),
             str(w_len),
             str(w_count)]))
    # print("samples: " + str(time.time() - strt))
