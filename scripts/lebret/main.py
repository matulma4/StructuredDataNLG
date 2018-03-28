import keras.backend as K
from keras.layers import Embedding, Input, Dense, Activation, Lambda, concatenate, dot, add, Flatten
from keras.models import Model

from config import *


def create_model(loc_dim, glob_dim):
    in_len = 10
    cond_len = 10
    c_input = Input(shape=(in_len,))

    ls_input = Input(shape=(cond_len,))
    le_input = Input(shape=(cond_len,))

    gf_input = Input(shape=(in_len,))
    gv_input = Input(shape=(in_len,))

    mix_input = Input(shape=(V,))

    con_size = n - 1

    context = Embedding(input_dim=V, output_dim=d, input_length=1)(c_input)
    flat_context = Flatten()(context)

    # loc_dim: number of fields x number of positions
    local_start = Embedding(input_dim=loc_dim, output_dim=d, input_length=l, mask_zero=True)(ls_input)
    local_end = Embedding(input_dim=loc_dim, output_dim=d, input_length=l, mask_zero=True)(le_input)
    ls_lambda = Lambda(lambda x: K.max(x, axis=1))(local_start)
    le_lambda = Lambda(lambda x: K.max(x, axis=1))(local_end)
    global_field = Embedding(input_dim=glob_dim, output_dim=g, input_length=l)(gf_input)
    global_value = Embedding(input_dim=glob_dim, output_dim=g, input_length=l)(gv_input)
    gf_lambda = Lambda(lambda x: K.max(x, axis=1))(global_field)
    gv_lambda = Lambda(lambda x: K.max(x, axis=1))(global_value)

    merged = concatenate([flat_context, ls_lambda, le_lambda, gf_lambda, gv_lambda])

    first = Dense(units=nhu, activation='tanh')(merged)  # h(x), 256
    second = Dense(units=V)(first)  # 20000

    # Mixing outputs
    mix = Embedding(input_dim=loc_dim, output_dim=d, input_length=V)(mix_input)  # 20000 x  x d
    third = Dense(units=nhu, activation='tanh')(mix)
    max_ftr = Lambda(lambda x: K.max(x, axis=1))(third)
    dot_prod = dot([max_ftr, first], axes=1)
    final = add([second, dot_prod])
    activate = Activation('softmax')(final)

    model = Model(inputs=[c_input, ls_input, le_input, gf_input, gv_input, mix_input], outputs=activate)
    model.compile(optimizer='sgd', loss='binary_crossentropy')
    return model


if __name__ == '__main__':
    # embed_table()
    create_model(100, 100)
