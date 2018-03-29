import pickle

import keras.backend as K
from keras.layers import Embedding, Input, Dense, Lambda, concatenate, Flatten
from keras.models import Model

from config import *


# TODO get actual sizes: vocabulary (V)
# TODO Maximum fields in table (glob_field_dim)
# TODO Maximum words in table (glob_word_dim)
# TODO Maximum occurrences of words in field (loc_dim)
# TODO Maximum index of local embeddings (max_loc_idx)
def create_model(loc_dim, glob_field_dim, glob_word_dim, max_loc_idx, max_glob_field_idx, max_glob_word_idx):
    in_len = 10
    cond_len = 10
    c_input = Input(shape=(in_len,), name='c_input')

    ls_input = Input(shape=(cond_len, loc_dim), name='ls_input')
    le_input = Input(shape=(cond_len, loc_dim), name='le_input')

    gf_input = Input(shape=(glob_field_dim,), name='gf_input')
    gv_input = Input(shape=(glob_word_dim,), name='gw_input')

    mix_input = Input(shape=(V,), name='mix_input')

    context = Embedding(input_dim=V, output_dim=d, input_length=1)(c_input)
    flat_context = Flatten()(context)

    # loc_dim: number of fields x number of positions
    local_start = Embedding(input_dim=max_loc_idx, output_dim=d, input_length=l, mask_zero=True)(ls_input)
    local_end = Embedding(input_dim=max_loc_idx, output_dim=d, input_length=l, mask_zero=True)(le_input)
    ls_lambda = Lambda(lambda x: K.max(x, axis=2))(local_start)
    le_lambda = Lambda(lambda x: K.max(x, axis=2))(local_end)
    flat_ls = Flatten()(ls_lambda)
    flat_le = Flatten()(le_lambda)
    global_field = Embedding(input_dim=max_glob_field_idx, output_dim=g, input_length=l)(gf_input)
    global_value = Embedding(input_dim=max_glob_word_idx, output_dim=g, input_length=l)(gv_input)
    gf_lambda = Lambda(lambda x: K.max(x, axis=1))(global_field)
    gv_lambda = Lambda(lambda x: K.max(x, axis=1))(global_value)

    merged = concatenate([flat_context, flat_ls, flat_le, gf_lambda, gv_lambda])

    first = Dense(units=nhu, activation='tanh', input_dim=2176)(merged)  # h(x), 256
    second = Dense(units=V, name='second')(first)  # 20000

    # Mixing outputs
    # mix = Embedding(input_dim=loc_dim, output_dim=d, input_length=V)(mix_input)  # 20000 x  x d
    # third = Dense(units=nhu, activation='tanh')(mix)
    # max_ftr = Lambda(lambda x: K.max(x, axis=1))(third)
    # dot_prod = dot([max_ftr, first], axes=1)
    # final = add([second, dot_prod])
    # activate = Activation('softmax')(final)

    model = Model(inputs=[c_input, ls_input, le_input, gf_input, gv_input], outputs=second)
    model.compile(optimizer='sgd', loss='binary_crossentropy')
    return model


if __name__ == '__main__':
    global V
    with open("params.txt") as f:
        V, max_loc_idx, glob_field_dim, glob_word_dim, loc_dim, f_len, w_len = [int(a) for a in f.read().split()]

    samples_context, samples_ls, samples_le, samples_gf, samples_gw, target = pickle.load(open("samples.pickle", "rb"))
    V = target.shape[1]
    model = create_model(loc_dim, f_len, w_len, max_loc_idx, glob_field_dim, glob_word_dim)
    # samples_context = np.array(samples_context)
    # samples_ls = np.array(samples_ls)
    # samples_le = np.array(samples_le)
    # samples_gf = np.array(samples_gf)
    # samples_gw = np.array(samples_gw)
    model.fit({'c_input': samples_context, 'ls_input': samples_ls, 'le_input': samples_le,
               'gf_input': samples_gf,
               'gw_input': samples_gw}, {'second': target}, batch_size=32)
