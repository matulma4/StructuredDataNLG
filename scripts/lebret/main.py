import inspect
import os
import sys

import keras.backend as K
import numpy as np
import pickle
from keras.layers import Embedding, Input, Dense, Lambda, concatenate, Flatten, Activation, dot, add, multiply
from keras.models import Model
from keras.optimizers import SGD

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from config import *

batch_len = 1


def reset(k):
    return [[] for _ in range(k)]


def reset_none(k, r):
	"""
	Create a list of lists

    Keyword arguments:
    k -- number of lists
    r -- size of each list
    """
    return [[None for _ in range(r)] for _ in range(k)]


def keras_log_likelihood(y_true, y_pred):
	"""
	Custom loss function used in training

    Keyword arguments:
    y_true -- gold standard
    y_pred -- neural network prediction
    """
    mult = multiply([y_true, y_pred])
    sm = K.sum(mult, axis=1)
    var = K.ones(shape=[batch_len]) * 1e-5
    res = -K.sum(K.log(add([sm, var])))
    return res


def log_likelihood(y_true, y_pred):
	"""
	Emulation of custom loss function used in training

    Keyword arguments:
    y_true -- gold standard
    y_pred -- neural network prediction
    """
    mult = np.multiply(y_true, y_pred)
    sm = np.sum(mult, axis=1)
    z = 1e-5 * np.ones(sm.shape[0])
    res = -np.sum(np.log(sm + z))
    return res


def create_model(loc_dim, glob_field_dim, glob_word_dim, max_loc_idx, max_glob_field_idx, max_glob_word_idx):
	"""
	Construct a neural network architecture

    """
    c_input = Input(shape=(l,), name='c_input')
    if use_ft:
        path_to_files = path + "pickle/" + dataset
        vectors = pickle.load(open(path_to_files + "/vectors.pickle", "rb"))
        if vectors is not None:
            context = Embedding(input_dim=V + 2, output_dim=d, input_length=l, weights=[vectors])(c_input)
        else:
            context = Embedding(input_dim=V + 2, output_dim=d, input_length=l)(c_input)
    else:
        context = Embedding(input_dim=V + 2, output_dim=d, input_length=l)(c_input)
    flat_context = Flatten()(context)

    emb_list = [flat_context]
    input_list = [c_input]
    emb_dim = l * d

    if local_cond:
        ls_input = Input(shape=(l, loc_dim,), name='ls_input')
        le_input = Input(shape=(l, loc_dim,), name='le_input')
        local_start = Embedding(input_dim=max_loc_idx + 1, output_dim=d, input_length=l, mask_zero=True)(ls_input)
        local_end = Embedding(input_dim=max_loc_idx + 1, output_dim=d, input_length=l, mask_zero=True)(le_input)
        ls_lambda = Lambda(lambda x: K.max(x, axis=2), output_shape=(l, d))(local_start)
        le_lambda = Lambda(lambda x: K.max(x, axis=2), output_shape=(l, d))(local_end)
        flat_ls = Flatten(input_shape=(l, d))(ls_lambda)
        flat_le = Flatten(input_shape=(l, d))(le_lambda)
        emb_list += [flat_ls, flat_le]
        input_list += [ls_input, le_input]
        emb_dim += 2 * l * d

    if global_cond:
        gf_input = Input(shape=(glob_field_dim,), name='gf_input')
        gv_input = Input(shape=(glob_word_dim,), name='gw_input')
        global_field = Embedding(input_dim=max_glob_field_idx, output_dim=g, input_length=l, mask_zero=True)(gf_input)
        global_value = Embedding(input_dim=max_glob_word_idx, output_dim=g, input_length=l, mask_zero=True)(gv_input)
        gf_lambda = Lambda(lambda x: K.max(x, axis=1))(global_field)
        gv_lambda = Lambda(lambda x: K.max(x, axis=1))(global_value)
        emb_list += [gf_lambda, gv_lambda]
        input_list += [gf_input, gv_input]
        emb_dim += 2 * g

    # loc_dim: number of fields x number of positions
    if global_cond or local_cond:
        merged = concatenate(emb_list)
    else:
        merged = emb_list[0]

    first = Dense(units=nhu, activation='tanh', input_dim=emb_dim, name='first')(merged)  # h(x), 256
    second = Dense(units=O, name='second')(first)  # 20000

    # Mixing outputs
    if use_mix:
        mix_input = Input(shape=(w_count, loc_dim,), name='mix_input')
        input_list.append(mix_input)
        mix = Embedding(input_dim=max_loc_idx, output_dim=d, input_length=w_count * loc_dim, mask_zero=True)(
            mix_input)  # 20000 x  x d
        third = Dense(units=nhu, activation='tanh', name='third')(mix)
        max_ftr = Lambda(lambda x: K.max(x, axis=1))(third)
        dot_prod = dot([max_ftr, first], axes=1)
        final = add([second, dot_prod])
        activate = Activation('softmax', name='activation')(final)
    else:
        activate = Activation('softmax', name='activation')(second)

    model = Model(inputs=input_list, outputs=activate)
    optimizer = SGD(lr=alpha, decay=decay_rate)
    model.compile(optimizer=optimizer, loss=keras_log_likelihood)  # 'categorical_crossentropy')
    return model


def create_one_sample(idx, s, e, bi, ei, max_l):
	"""
	Create a sample for training; includes context and local conditioning

    """
    context = np.array(idx[bi:ei])
    s_context = np.array([np.pad(ss, (0, max_l - len(ss)), mode='constant') for ss in s[bi:ei]])
    e_context = np.array([np.pad(ee, (0, max_l - len(ee)), mode='constant') for ee in e[bi:ei]])
    return context, e_context, s_context


def dump_garbage():
    """
    show us what's the garbage about
    """
    ffs = 150
    # force collection
    print("\nGARBAGE:")
    gc.collect()

    print("\nGARBAGE OBJECTS:")
    for x in gc.garbage:
        s = str(gc.get_referrers(x))
        if len(s) > ffs: s = s[:ffs]
        print(str(x)[:ffs], "\n  ", s)


# TODO make global conditioning more effective
def create_samples(indices, start, end, t_f, t_w, fields, max_l, output, sentences):
	"""
	Create samples and train the model on them

    """
    samplecount = 0
    inputs, outputs = reset_none(2, sample_limit)
    for i in range(len(indices)):
        idx = indices[i]
        s = start[i]
        e = end[i]
        samples_context, samples_ls, samples_le, samples_gf, samples_gw, samples_mix, target = reset_none(7,
                                                                                                          len(idx) - l)
        if global_cond:
            glob_field = np.pad(t_f[i], (0, f_len - len(t_f[i])), mode='constant')
            glob_word = np.pad(t_w[i], (0, w_len - len(t_w[i])), mode='constant')

        field = fields[i]
        if use_mix:
            mix_sample = []
            for t_key in field:
                vt = np.unique([tv[0] * l + tv[1] for tv in field[t_key][:max_l]])
                mix_sample.append(np.pad(vt, (0, max_l - vt.shape[0]), mode='constant'))
            mix_sample = np.pad(np.array(mix_sample), ((0, w_count - len(mix_sample)), (0, 0)), mode='constant')

        for j in range(l, len(idx)):
            context, s_context, e_context = create_one_sample(idx, s, e, j - l, j, max_l)
            samples_context[j - l] = context
            if local_cond:
                samples_ls[j - l] = s_context
                samples_le[j - l] = e_context
            if global_cond:
                samples_gf[j - l] = glob_field
                samples_gw[j - l] = glob_word
            if use_mix:
                samples_mix[j - l] = mix_sample
            t = np.zeros(len(output) + 1)
            try:
                t[np.where(output == sentences[i][j - l])[0][0]] = 1.0
            except IndexError:
                t[-1] = 1.0
            target[j - l] = t
        global batch_len
        batch_len = len(samples_context)
        input_ls = {'c_input': np.array(samples_context)}
        if local_cond:
            input_ls['ls_input'] = np.array(samples_ls)
            input_ls['le_input'] = np.array(samples_le)
        if global_cond:
            input_ls['gf_input'] = np.array(samples_gf)
            input_ls['gw_input'] = np.array(samples_gw)
        if use_mix:
            input_ls['mix_input'] = np.array(samples_mix)
        inputs[samplecount] = input_ls
        outputs[samplecount] = target
        samplecount += 1
        if samplecount == sample_limit:
            for it in range(n_iter):
                for ex in range(sample_limit):
                    #
                    loss = model.train_on_batch(inputs[ex], {'activation': np.array(outputs[ex])})
                    print("Training epoch " + str(it) + " on " + str(len(outputs[ex])) + " samples, loss: " + str(loss))
                model.save(path + "models/" + dataset + "/" + hashed + ".h5")

            inputs, outputs = reset_none(2, sample_limit)
            samplecount = 0
            gc.collect()
    for it in range(n_iter):
        for ex in range(samplecount):
            loss = model.train_on_batch(inputs[ex], {'activation': np.array(outputs[ex])})
            print("Training epoch " + str(it) + " on " + str(len(outputs[ex])) + " samples, loss: " + str(
                loss))
		model.save(path + "models/" + dataset + "/" + hashed + ".h5")



def load_from_file(hashed):
	"""
	Load data from file

    Keyword arguments:
    hashed -- name of the file
    """
    path_to_files = path + "pickle/" + dataset + "/" + hashed
    output = pickle.load(open(path_to_files + "/output.pickle", "rb"))
    start = pickle.load(open(path_to_files + "/start.pickle", "rb"))
    end = pickle.load(open(path_to_files + "/end.pickle", "rb"))
    indices = pickle.load(open(path_to_files + "/indices.pickle", "rb"))
    t_fields = pickle.load(open(path_to_files + "/t_fields.pickle", "rb"))
    t_words = pickle.load(open(path_to_files + "/t_words.pickle", "rb"))
    infoboxes = pickle.load(open(path_to_files + "/infoboxes.pickle", "rb"))
    with open(path_to_files + "/sents.dlx", "r", encoding="utf-8") as g:
        sentences = [line.strip().split() for line in g]
    return indices, start, end, t_fields, t_words, infoboxes, output, sentences


if __name__ == '__main__':
    gc.enable()
    global V, n_iter, l, use_ft
    h = sys.argv[1]
    l = int(h[8:10])
    use_ft = bool(int(h[13]))
    hashed = h + str(n_iter).zfill(3) + "".join([str(int(boole)) for boole in [local_cond, global_cond, use_mix]])
    alpha = 0.025
    with open(path + "pickle/" + dataset + "/" + h + "/params.txt") as f:
        V, max_loc_idx, glob_field_dim, glob_word_dim, loc_dim, f_len, w_len, w_count = [int(a) for a in
                                                                                         f.read().split()]

    indices, start, end, t_fields, t_words, infoboxes, output, sentences = load_from_file(h)
    O = output.shape[0] + 1
    model = create_model(loc_dim, f_len, w_len, max_loc_idx, glob_field_dim + 1, glob_word_dim + 1)
    create_samples(indices, start, end, t_fields, t_words, infoboxes, loc_dim, output, sentences)
