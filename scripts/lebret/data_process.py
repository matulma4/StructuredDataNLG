import pickle
import re
from collections import Counter

import numpy as np
from sklearn.preprocessing import LabelEncoder

from config import *
from data_loader import load_sentences, load_infoboxes


def get_words(sentences):
    vocabulary = get_most_frequent(sentences, True)
    sents = replace_oov(sentences, vocabulary)
    indices, encoder, max_idx = transform_to_indices(sents)
    return indices, encoder, max_idx


def local_conditioning(tf, f, sentences):
    F = len(tf)
    start = []
    end = []
    max_len = 0
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
    values = np.array([b for a in data for b in a])
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    i = 0
    result = []
    for d in data:
        result.append(integer_encoded[i:i + len(d)])
        i += len(d)
    return result, label_encoder, max(integer_encoded)


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
    fields = []
    le = LabelEncoder()
    tf = dict(zip(unique_keys, le.fit_transform(unique_keys)))
    table_fields = []
    table_words = []

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
                table_f.add(tf[k])

                if r[key] in field:
                    field[word].append((tf[k], min(l, idx), max(1, l - idx + 1)))
                else:
                    field[word] = [(tf[k], min(l, idx), max(1, l - idx + 1))]
            if word in encoder.classes_:
                table_w.add(word)
        table_fields.append(list(table_f))
        table_words.append(encoder.transform(list(table_w)))
        f_size = max(f_size, max(table_fields[-1]))
        w_size = max(w_size, max(table_words[-1]))
        f_len = max(f_len, len(table_fields[-1]))
        w_len = max(w_len, len(table_words[-1]))
        w_count = max(w_count, len(field))
        fields.append(field)
    print("Maximum fields in table: " + str(f_len))
    print("Maximum words in table: " + str(w_len))
    return fields, tf, table_fields, table_words


def delexicalize(sentences, tables, vocabulary, keys):
    field_names = set()
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
    return sentences, list(field_names)


# TODO make create_one_sample method
def create_samples(indices, start, end, t_f, t_w, fields):
    samples_context = []
    samples_ls = []
    samples_le = []
    samples_gf = []
    samples_gw = []
    samples_mix = []
    target = []
    filecount = 0
    samplecount = 0
    for i in range(len(indices)):
        idx = indices[i]
        s = start[i]
        e = end[i]
        glob_field = np.pad(t_f[i], (0, f_len - len(t_f[i])), mode='constant')
        glob_word = np.pad(t_w[i], (0, w_len - len(t_w[i])), mode='constant')
        field = fields[i]
        for j in range(l, len(idx)):
            context = idx[j - l:j]
            s_context = np.array([np.pad(ss, (0, max_l - len(ss)), mode='constant') for ss in s[j - l:j]])
            e_context = np.array([np.pad(ee, (0, max_l - len(ee)), mode='constant') for ee in e[j - l:j]])
            samples_context.append(context)
            samples_ls.append(s_context)
            samples_le.append(e_context)
            samples_gf.append(glob_field)
            samples_gw.append(glob_word)
            t = np.zeros(len(output) + 1)
            try:
                xx = np.where(output == sentences[i][j])
                t[xx[0][0]] = 1.0
            except IndexError:
                t[-1] = 1.0
            target.append(t)
            samplecount += 1
            if samplecount == sample_limit:
                pickle.dump((np.array(samples_context), np.array(samples_ls), np.array(samples_le), np.array(samples_gf), np.array(samples_gw), np.array(target)),
                open(path + "samples/" + dataset + "/samples_" + str(filecount) + ".pickle", "wb"))
                samples_context = []
                samples_ls = []
                samples_le = []
                samples_gf = []
                samples_gw = []
                target = []
                samplecount = 0
                filecount += 1

    pickle.dump((samples_context, samples_ls, samples_le, samples_gf, samples_gw, target),
                            open(path + "samples/" + dataset + "/samples_" + str(filecount) + ".pickle", "wb"))


if __name__ == '__main__':
    dicts, u_keys = load_infoboxes(data_path, dataset)
    sentences = load_sentences()
    f_size = 0
    w_size = 0
    f_len = 0
    w_len = 0
    w_count = 0
    indices, encoder, max_idx = get_words(sentences)
    f, tf, t_fields, t_words = process_infoboxes(u_keys, dicts, encoder)
    print("Size of vocabulary: " + str(max_idx))
    start, end, max_l = local_conditioning(tf, f, sentences)
    print("Number of fields: " + str(max_l))
    sentences, f_names = delexicalize(sentences, dicts, encoder.classes_, u_keys)
    output = np.concatenate((encoder.classes_, f_names))
    with open(path + "samples/" + dataset + "/params.txt", "w") as g:
        g.write(" ".join(
            [str(max_idx), str(len(tf) * l + 2), str(f_size), str(w_size), str(max_l), str(f_len), str(w_len),
             str(w_count)]))
    create_samples(indices, start, end, t_fields, t_words, f)
