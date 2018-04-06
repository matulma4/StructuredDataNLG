import inspect

import keras.backend as K
import numpy as np
import pickle
import sys
from keras.layers import Embedding, Input, Dense, Lambda, concatenate, Flatten, Activation
from keras.models import Model
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from config import *


def create_model(loc_dim, glob_field_dim, glob_word_dim, max_loc_idx, max_glob_field_idx, max_glob_word_idx):
    c_input = Input(shape=(l,), name='c_input')

    ls_input = Input(shape=(l, loc_dim,), name='ls_input')
    le_input = Input(shape=(l, loc_dim,), name='le_input')

    gf_input = Input(shape=(glob_field_dim,), name='gf_input')
    gv_input = Input(shape=(glob_word_dim,), name='gw_input')

    mix_input = Input(shape=(w_count, loc_dim,), name='mix_input')

    context = Embedding(input_dim=V, output_dim=d, input_length=l)(c_input)
    flat_context = Flatten()(context)

    # loc_dim: number of fields x number of positions
    local_start = Embedding(input_dim=max_loc_idx, output_dim=d, input_length=l, mask_zero=True)(ls_input)
    local_end = Embedding(input_dim=max_loc_idx, output_dim=d, input_length=l, mask_zero=True)(le_input)
    ls_lambda = Lambda(lambda x: K.max(x, axis=2), output_shape=(l, d))(local_start)
    le_lambda = Lambda(lambda x: K.max(x, axis=2), output_shape=(l, d))(local_end)
    flat_ls = Flatten(input_shape=(l, d))(ls_lambda)
    flat_le = Flatten(input_shape=(l, d))(le_lambda)
    global_field = Embedding(input_dim=max_glob_field_idx, output_dim=g, input_length=l, mask_zero=True)(gf_input)
    global_value = Embedding(input_dim=max_glob_word_idx, output_dim=g, input_length=l, mask_zero=True)(gv_input)
    gf_lambda = Lambda(lambda x: K.max(x, axis=1))(global_field)
    gv_lambda = Lambda(lambda x: K.max(x, axis=1))(global_value)

    merged = concatenate([flat_context, flat_ls, flat_le, gf_lambda, gv_lambda])

    first = Dense(units=nhu, activation='tanh', input_dim=3 * l * d + 2 * g)(merged)  # h(x), 256
    second = Dense(units=V, name='second')(first)  # 20000

    # Mixing outputs
    # mix = Embedding(input_dim=loc_dim, output_dim=d, input_length=V)(mix_input)  # 20000 x  x d
    # third = Dense(units=nhu, activation='tanh')(mix)
    # max_ftr = Lambda(lambda x: K.max(x, axis=1))(third)
    # dot_prod = dot([max_ftr, first], axes=1)
    # final = add([second, dot_prod])
    activate = Activation('softmax', name='activation')(second)

    model = Model(inputs=[c_input, ls_input, le_input, gf_input, gv_input], outputs=activate)
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    return model


def create_one_sample(idx, s, e, j, max_l):
    context = idx[j - l:j]
    s_context = np.array([np.pad(ss, (0, max_l - len(ss)), mode='constant') for ss in s[j - l:j]])
    e_context = np.array([np.pad(ee, (0, max_l - len(ee)), mode='constant') for ee in e[j - l:j]])
    return context, e_context, s_context


# TODO make global conditioning more effective
def create_samples(indices, start, end, t_f, t_w, fields, max_l, output, sentences):
    samples_context = []
    samples_ls = []
    samples_le = []
    samples_gf = []
    samples_gw = []
    samples_mix = []
    target = []
    # filecount = 0
    samplecount = 0
    for i in range(len(indices)):
        idx = indices[i]
        s = start[i]
        e = end[i]
        glob_field = np.pad(t_f[i], (0, f_len - len(t_f[i])), mode='constant')
        glob_word = np.pad(t_w[i], (0, w_len - len(t_w[i])), mode='constant')
        field = fields[i]
        # mix_sample = []
        # for t_key in field:
        #     vt = np.unique([tv[0] * l + tv[1] for tv in field[t_key]])
        #     mix_sample.append(np.pad(vt, (0, max_l - vt.shape[0]), mode='constant'))
        # mix_sample = np.pad(np.array(mix_sample), ((0, w_count - len(mix_sample)), (0, 0)), mode='constant')
        for j in range(l, len(idx)):
            context, s_context, e_context = create_one_sample(idx, s, e, j, max_l)
            samples_context.append(context)
            samples_ls.append(s_context)
            samples_le.append(e_context)
            samples_gf.append(glob_field)
            samples_gw.append(glob_word)
            # samples_mix.append(mix_sample)
            t = np.zeros(len(output) + 1)
            try:
                xx = np.where(output == sentences[i][j])
                t[xx[0][0]] = 1.0
            except IndexError:
                t[-1] = 1.0
            target.append(t)
            samplecount += 1
            if samplecount == sample_limit:
                model.train_on_batch({'c_input': np.array(samples_context), 'ls_input': np.array(samples_ls), 'le_input': np.array(samples_le),
                                      'gf_input': np.array(samples_gf),
                                      'gw_input': np.array(samples_gw)}, {'activation': np.array(target)})
                # pickle.dump((
                #     np.array(samples_context), np.array(samples_ls), np.array(samples_le), np.array(samples_gf),
                #     np.array(samples_gw), np.array(samples_mix), np.array(target)),
                #     open(path + "samples/" + dataset + "/samples_" + str(filecount) + ".pickle", "wb"))
                samples_context = []
                samples_ls = []
                samples_le = []
                samples_gf = []
                samples_gw = []
                samples_mix = []
                target = []
                samplecount = 0
                # filecount += 1
    # pickle.dump((
    #     np.array(samples_context), np.array(samples_ls), np.array(samples_le), np.array(samples_gf),
    #     np.array(samples_gw), np.array(samples_mix), np.array(target)),
    #     open(path + "samples/" + dataset + "/samples_" + str(filecount) + ".pickle", "wb"))


def load_from_file():
    path_to_files = path + "pickle/" + dataset
    output = pickle.load(open(path_to_files + "/output.pickle", "rb"))
    start = pickle.load(open(path_to_files + "/start.pickle", "rb"))
    end = pickle.load(open(path_to_files + "/end.pickle", "rb"))
    indices = pickle.load(open(path_to_files + "/indices.pickle", "rb"))
    t_fields = pickle.load(open(path_to_files + "/t_fields.pickle", "rb"))
    t_words = pickle.load(open(path_to_files + "/t_words.pickle", "rb"))
    infoboxes = pickle.load(open(path_to_files + "/infoboxes.pickle", "rb"))
    with open(data_path + "/" + dataset + ".dlx", "r", encoding="utf-8") as g:
        sentences = [line.strip().split() for line in g]
    return indices, start, end, t_fields, t_words, infoboxes, output, sentences


if __name__ == '__main__':
    n_iter = 10
    global V
    with open(path + "samples/" + dataset + "/params.txt") as f:
        V, max_loc_idx, glob_field_dim, glob_word_dim, loc_dim, f_len, w_len, w_count = [int(a) for a in
                                                                                         f.read().split()]

    indices, start, end, t_fields, t_words, infoboxes, output, sentences = load_from_file()
    V = output.shape[0]+1
    model = create_model(loc_dim, f_len, w_len, max_loc_idx, glob_field_dim + 1, glob_word_dim + 1)
    for it in range(n_iter):
        create_samples(indices, start, end, t_fields, t_words, infoboxes, loc_dim, output, sentences)
    model.save(path + "models/" + dataset + "/model_" + str(iter) + ".h5")

    # samples_context, samples_ls, samples_le, samples_gf, samples_gw, samples_mix, target = pickle.load(
    #     open(path + "samples/" + dataset + "/samples_0.pickle", "rb"))
    # V = target.shape[1]

