import inspect

import keras.backend as K
import numpy as np
import pickle
import sys
from keras.layers import Embedding, Input, Dense, Lambda, concatenate, Flatten, Activation, dot, add, multiply
from keras.models import Model
import os, gc

from keras.optimizers import SGD

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from config import *

batch_len = 1


def reset(k):
    return [[] for _ in range(k)]


def keras_log_likelihood(y_true, y_pred):
    # tf_session = K.get_session()
    mult = multiply([y_true, y_pred])
    sm = K.sum(mult, axis=1)
    var = K.ones(shape=[batch_len])*1e-5
    res = -K.sum(K.log(add([sm, var])))
    return res


def log_likelihood(y_true, y_pred):
    mult = np.multiply(y_true, y_pred)
    sm = np.sum(mult, axis=1)
    z = 1e-5 * np.ones(sm.shape[0])
    res = -np.sum(np.log(sm + z))
    return res


def create_model(loc_dim, glob_field_dim, glob_word_dim, max_loc_idx, max_glob_field_idx, max_glob_word_idx):
    c_input = Input(shape=(l,), name='c_input')
    if use_ft:
        path_to_files = path + "pickle/" + dataset
        vectors = pickle.load(open(path_to_files + "/vectors.pickle", "rb"))
        if vectors is not None:
            context = Embedding(input_dim=V+2, output_dim=d, input_length=l, weights=[vectors])(c_input)
        else:
            context = Embedding(input_dim=V+2, output_dim=d, input_length=l)(c_input)
    else:
        context = Embedding(input_dim=V+2, output_dim=d, input_length=l)(c_input)
    flat_context = Flatten()(context)

    emb_list = [flat_context]
    input_list = [c_input]
    emb_dim = l * d

    if local_cond:
        ls_input = Input(shape=(l, loc_dim,), name='ls_input')
        le_input = Input(shape=(l, loc_dim,), name='le_input')
        local_start = Embedding(input_dim=max_loc_idx+1, output_dim=d, input_length=l, mask_zero=True)(ls_input)
        local_end = Embedding(input_dim=max_loc_idx+1, output_dim=d, input_length=l, mask_zero=True)(le_input)
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
        mix = Embedding(input_dim=max_loc_idx, output_dim=d, input_length=w_count*loc_dim, mask_zero=True)(mix_input)  # 20000 x  x d
        third = Dense(units=nhu, activation='tanh',name='third')(mix)
        max_ftr = Lambda(lambda x: K.max(x, axis=1))(third)
        dot_prod = dot([max_ftr, first], axes=1)
        final = add([second, dot_prod])
        activate = Activation('softmax', name='activation')(final)
    else:
        activate = Activation('softmax', name='activation')(second)


    model = Model(inputs=input_list, outputs=activate)
    optimizer = SGD(lr=0.025, decay=decay_rate)
    model.compile(optimizer=optimizer, loss=keras_log_likelihood)#'categorical_crossentropy')
    return model


def create_one_sample(idx, s, e, bi, ei, max_l):
    context = np.array(idx[bi:ei])
    s_context = np.array([np.pad(ss, (0, max_l - len(ss)), mode='constant') for ss in s[bi:ei]])
    e_context = np.array([np.pad(ee, (0, max_l - len(ee)), mode='constant') for ee in e[bi:ei]])
    return context, e_context, s_context


# TODO make global conditioning more effective
def create_samples(indices, start, end, t_f, t_w, fields, max_l, output, sentences):
    samples_context, samples_ls, samples_le, samples_gf, samples_gw, samples_mix, target = reset(7)
    # filecount = 0
    samplecount = 0
    inputs, outputs = reset(2)
    lr = model.optimizer.lr
    for i in range(len(indices)):
        idx = indices[i]
        s = start[i]
        e = end[i]
        if global_cond:
            glob_field = np.pad(t_f[i], (0, f_len - len(t_f[i])), mode='constant')
            glob_word = np.pad(t_w[i], (0, w_len - len(t_w[i])), mode='constant')

        field = fields[i]
        if use_mix:
            mix_sample = []
            for t_key in field:
                vt = np.unique([tv[0] * l + tv[1] for tv in field[t_key]])
                mix_sample.append(np.pad(vt, (0, max_l - vt.shape[0]), mode='constant'))
            mix_sample = np.pad(np.array(mix_sample), ((0, w_count - len(mix_sample)), (0, 0)), mode='constant')

        for j in range(l, len(idx)):
            context, s_context, e_context = create_one_sample(idx, s, e, j-l, j, max_l)
            samples_context.append(context)
            if local_cond:
                samples_ls.append(s_context)
                samples_le.append(e_context)
            if global_cond:
                samples_gf.append(glob_field)
                samples_gw.append(glob_word)
            if use_mix:
                samples_mix.append(mix_sample)
            t = np.zeros(len(output) + 1)
            try:
                xx = np.where(output == sentences[i][j-l])
                t[xx[0][0]] = 1.0
            except IndexError:
                t[-1] = 1.0
            target.append(t)
            samplecount += 1
            # if samplecount == sample_limit:
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
        inputs.append(input_ls)
        outputs.append(target)
        samples_context, samples_ls, samples_le, samples_gf, samples_gw, samples_mix, target = reset(7)
        if len(inputs) == sample_limit:
            for it in range(n_iter):
                for ex in range(sample_limit):
                    loss = model.train_on_batch(inputs[ex], {'activation': np.array(outputs[ex])})
                    lr *= (1. / (1. + model.optimizer.decay * K.cast(model.optimizer.iterations, K.dtype(model.optimizer.decay))))
                    print("Training epoch " + str(it) + " on " + str(len(outputs[ex])) + " samples, loss: " + str(loss) + ", learning rate: " + str(K.eval(lr)))
                    model.save(path + "models/" + dataset + "/" + hashed + ".h5")
            inputs, outputs = reset(2)
            # print(pred)
            # pred = model.predict(x=input_ls)
            # print(log_likelihood(target, pred))
        samplecount = 0
        gc.collect()
    for it in range(n_iter):
        for ex in range(len(inputs)):
            loss = model.train_on_batch(inputs[ex], {'activation': np.array(outputs[ex])})
            lr *= (
            1. / (1. + model.optimizer.decay * K.cast(model.optimizer.iterations, K.dtype(model.optimizer.decay))))
            print("Training epoch " + str(it) + " on " + str(len(outputs[ex])) + " samples, loss: " + str(
                loss) + ", learning rate: " + str(K.eval(lr)))
            model.save(path + "models/" + dataset + "/" + hashed + ".h5")

    # input_ls = {'c_input': np.array(samples_context), 'mix_input' : np.array(samples_mix)}
    # if local_cond:
    #     input_ls['ls_input'] = np.array(samples_ls)
    #     input_ls['le_input'] = np.array(samples_le)
    # if global_cond:
    #     input_ls['gf_input'] = np.array(samples_gf)
    #     input_ls['gw_input'] = np.array(samples_gw)
    #
    # for it in range(n_iter):
    #     loss = model.train_on_batch(input_ls, {'activation': np.array(target)})
    #     lr *= (1. / (1. + model.optimizer.decay * K.cast(model.optimizer.iterations, K.dtype(model.optimizer.decay))))
    #     pred = np.argmax(model.predict(x=input_ls),axis=1)
    #     t = np.argmax(target, axis=1)
    #     acc = float(len(np.where(pred-t == 0)[0]))/float(len(target))*100.0
    #     print("Training epoch " + str(it) + " on " + str(samplecount) + " samples, loss: " + str(loss) + ", learning rate: " + str(K.eval(lr)) + ", Accuracy: " + str(acc))


def train(input_ls, target, lr, samplecount):
    for it in range(n_iter):
        loss = model.train_on_batch(input_ls, {'activation': np.array(target)})
        lr *= (1. / (1. + model.optimizer.decay * K.cast(model.optimizer.iterations, K.dtype(model.optimizer.decay))))
        print("Training epoch " + str(it) + " on " + str(samplecount) + " samples, loss: " + str(
            loss) + ", learning rate: " + str(K.eval(lr)))


def load_from_file(hashed):
    path_to_files = path + "pickle/" + dataset + "/" + hashed
    output = pickle.load(open(path_to_files + "/output.pickle", "rb"))
    start = pickle.load(open(path_to_files + "/start.pickle", "rb"))
    end = pickle.load(open(path_to_files + "/end.pickle", "rb"))
    indices = pickle.load(open(path_to_files + "/indices.pickle", "rb"))
    t_fields = pickle.load(open(path_to_files + "/t_fields.pickle", "rb"))
    t_words = pickle.load(open(path_to_files + "/t_words.pickle", "rb"))
    infoboxes = pickle.load(open(path_to_files + "/infoboxes.pickle", "rb"))
    with open(path + "/pickle/" + dataset + "/sents.dlx", "r", encoding="utf-8") as g:
        sentences = [line.strip().split() for line in g]
    return indices, start, end, t_fields, t_words, infoboxes, output, sentences


if __name__ == '__main__':
    global V, n_iter, l, use_ft
    h = sys.argv[1]
    l = int(h[8:10])
    n_iter = int(h[10:13])
    use_ft = bool(int(h[16]))
    hashed = h + "".join([str(int(boole)) for boole in [local_cond, global_cond, use_mix]])

    with open(path + "pickle/" + dataset + "/" + h + "/params.txt") as f:
        V, max_loc_idx, glob_field_dim, glob_word_dim, loc_dim, f_len, w_len, w_count = [int(a) for a in
                                                                                         f.read().split()]

    indices, start, end, t_fields, t_words, infoboxes, output, sentences = load_from_file(h)
    O = output.shape[0]+1
    model = create_model(loc_dim, f_len, w_len, max_loc_idx, glob_field_dim + 1, glob_word_dim + 1)
    # for it in range(n_iter):
    create_samples(indices, start, end, t_fields, t_words, infoboxes, loc_dim, output, sentences)
    model.save(path + "models/" + dataset + "/model_" + str(n_iter) + ".h5")

    # samples_context, samples_ls, samples_le, samples_gf, samples_gw, samples_mix, target = pickle.load(
    #     open(path + "samples/" + dataset + "/samples_0.pickle", "rb"))
    # V = target.shape[1]
