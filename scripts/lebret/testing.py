import inspect
import os
import sys

import numpy as np
import pickle
from keras.models import load_model
from nltk.translate.bleu_score import sentence_bleu
from sample import Sample
from main import create_one_sample, keras_log_likelihood
from data_loader import load_infoboxes, load_sentences
from nltk.translate.bleu_score import SmoothingFunction

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from config import *


def get_n_best(ls, n):
    """
    Obtain n highest elements in a list

    Keyword arguments:
    ls -- a list
    n -- number of elements to retrieve
    """
    arr = np.array(ls)
    return arr.argsort()[-n:][::-1]


def replace_fields(sent, ib):
    """
    Replace delexicalized token with a word

    Keyword arguments:
    sent -- a list of words
    ib -- dict containing infobox table
    """
    for w in range(len(sent)):
        word = sent[w]
        if word in ib.keys():
            sent[w] = ib[word]
    return sent


def load_from_file(test_set, model_name, h):
    """
    Load various data from file

    """
    path_to_files = path + "pickle/" + dataset + "/" + h
    output = np.append(pickle.load(open(path_to_files + "/output.pickle", "rb")), "<UNK>")
    infoboxes, u_keys = load_infoboxes(test_path + test_set, test_set)
    field_transform = pickle.load(open(path_to_files + "/field_tf.pickle", "rb"))
    word_transform = pickle.load(open(path_to_files + "/word_tf.pickle", "rb"))
    model = load_model(path + "models/" + dataset + "/" + model_name + ".h5",
                       custom_objects={"keras_log_likelihood": keras_log_likelihood})
    return infoboxes, output, model, field_transform, word_transform


def make_sample(b, loc_dim):
    """
    Create one sample of context words, local and global conditioning

    Keyword arguments:
    b -- object of Sample class
    loc_dim -- dimension of sample
    """
    context, s_context, e_context = create_one_sample(b.indexes, b.starts, b.ends, -l, None, loc_dim)
    return [context], [s_context], [e_context]


def process_infoboxes(dict_list, field_transform, word_transform):
    """
    Transform infobox elements from strings to ints

    Keyword arguments:
    dict_list -- a list of infoboxes
    field_transform -- dict mapping field names to indices
    word_transform -- dict mapping field values to indices
    """
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


def beam_search(model, size, sent_length, output, word_tf, field_tf, gf, gw, infobox, mix_sample, loc_dim, output_beam):
    """
    Decode a sentence using the model

    """
    beam = [
        Sample(0.0, ["s" + str(i) for i in range(l)], [word_tf["s" + str(i)] for i in range(l - 1)], word_tf, field_tf,
               [[len(field_tf) * l + 2] for i in range(l - 1)], [[len(field_tf) * l + 2] for i in range(l - 1)],
               infobox)]
    # init first sample
    while True:
        new_beam = []
        new_score = []
        for b in beam:
            if len(b.sentence) == sent_length:  # or b.sentence[-1] == '.':
                return b.sentence, b.score
                # predict for each element in beam
            samples_context, samples_ls, samples_le = make_sample(b, loc_dim)
            prediction = model.predict(
                {'c_input': np.array(samples_context), 'ls_input': np.array(samples_ls),
                 'le_input': np.array(samples_le),
                 'gf_input': np.array(gf),
                 'gw_input': np.array(gw),
                 'mix_input': np.array([mix_sample])
                 })
            new_beam += [(b, output[p]) for p in range(prediction.shape[1])]
            new_score += [np.log(prediction[0][p]) + b.score for p in range(prediction.shape[1])]
        best_scores = get_n_best(new_score, size)
        beam = [
            Sample(new_score[bs], new_beam[bs][0].sentence + [new_beam[bs][1]], new_beam[bs][0].indexes, word_tf,
                   field_tf, new_beam[bs][0].starts, new_beam[bs][0].ends,
                   infobox) for bs in best_scores
            ]

        if output_beam:
            for b in beam:
                print(b.sentence[l:], str(b.score))
            print("=======================================")


def global_conditioning(t_f, t_w, f_len, w_len):
    """
    Create a global conditioning sample

    """
    if f_len >= len(t_f):
        gf = [np.pad(t_f, (0, f_len - len(t_f)), mode='constant')]
    else:
        gf = np.array(t_f[:f_len])
    if w_len >= len(t_w):
        gw = [np.pad(t_w, (0, w_len - len(t_w)), mode='constant')]
    else:
        gw = np.array(t_w[:w_len])
    return gf, gw


def test_model(model, infoboxes, f_tf, w_tf, output):
    """
    Run beam search over all testing sentences

    """
    outfile = open(m_name + ".out", "w")
    if os.path.exists(path + "pickle/" + dataset + "/" + h + "/test_ib.pickle"):
        t_fields, t_words, ib = pickle.load(open(path + "pickle/" + dataset + "/" + h + "/test_ib.pickle", "rb"))
    else:
        t_fields, t_words, ib = process_infoboxes(infoboxes, f_tf, w_tf)
        pickle.dump((t_fields, t_words, ib), open(path + "pickle/" + dataset + "/" + h + "/test_ib.pickle", "wb"))
    generated = []
    scores = []
    for i in range(len(ib)):
        mix_sample = []
        for t_key in ib[i]:
            vt = np.unique([tv[0] * l + tv[1] for tv in ib[i][t_key]])
            mix_sample.append(np.pad(vt, (0, loc_dim - vt.shape[0]), mode='constant'))
        mix_sample = np.pad(np.array(mix_sample), ((0, w_count - len(mix_sample)), (0, 0)), mode='constant')
        gf, gw = global_conditioning(t_fields[i], t_words[i], f_len, w_len)
        gen, score = beam_search(model, 10, 20 + l, output, w_tf, f_tf, gf, gw, ib[i], mix_sample, loc_dim, output_beam)
        rep = replace_fields(gen[l:], infoboxes[i])
        outfile.write(" ".join(rep) + "\n")
        generated.append(rep)
        scores.append(np.power(1.0 / np.exp(score), 1.0 / float(len(gen))))
    outfile.close()
    return generated, scores


if __name__ == '__main__':
    output_beam = False
    m_name = sys.argv[1]
    metric = "bleu"
    global l
    l = int(m_name[8:10])
    test_set = "test_r"
    h = m_name[:-6]

    # Load data
    infoboxes, output, model, field_transform, word_transform = load_from_file(test_set, m_name, h)
    with open(path + "pickle/" + dataset + "/" + h + "/params.txt") as f:
        V, max_loc_idx, glob_field_dim, glob_word_dim, loc_dim, f_len, w_len, w_count = [int(a) for a in
                                                                                         f.read().split()]

    # Generate sentences
    gen_sents, gen_scores = test_model(model, infoboxes[:1000], field_transform, word_transform, output)

    # Compute BLEU and perplexity
    sents = load_sentences(test_path + test_set + "/", test_set)
    bleu = []
    smooth = SmoothingFunction().method7
    for pred, true in zip(gen_sents, sents):
        bleu.append(sentence_bleu([true], pred, smoothing_function=smooth))
    print("BLEU: ", np.mean(bleu), "Perplexity: ", np.mean(gen_scores))
