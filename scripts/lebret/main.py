import os

from keras.layers import Embedding, Input, Dense, Activation, Lambda, concatenate, dot, add
from keras.models import Model

from data_loader import load_infoboxes, get_most_frequent
import numpy as np

d = 64
g = 128
l = 10
n = 11
V = 20000
nhu = 256


def embed_table():
    r, f = load_infoboxes(path, dataset)
    # TODO Get fields occurring at least 100 times
    result = get_most_frequent(r, False)

    f = len("")

    pass


def create_model(loc_dim, glob_dim):
    in_len = 1
    c_input = Input(shape=(in_len,))
    ls_input = Input(shape=(in_len,))
    le_input = Input(shape=(in_len,))
    gf_input = Input(shape=(in_len,))
    gv_input = Input(shape=(in_len,))
    mix_input = Input(shape=(V,))

    con_size = n - 1

    context = Embedding(input_dim=V, output_dim=con_size * d, input_length=l)(c_input)
    local_start = Embedding(input_dim=loc_dim, output_dim=con_size * d, input_length=l)(ls_input)
    local_end = Embedding(input_dim=loc_dim, output_dim=con_size * d, input_length=l)(le_input)
    global_field = Embedding(input_dim=glob_dim, output_dim=g, input_length=l)(gf_input)
    global_value = Embedding(input_dim=glob_dim, output_dim=g, input_length=l)(gv_input)

    merged = concatenate([context, local_start, local_end, global_field, global_value])

    first = Dense(units=nhu, activation='tanh')(merged)  # h(x), 256
    second = Dense(units=V)(first)  # 20000

    # Mixing outputs
    mix = Embedding(input_dim=loc_dim, output_dim=d, input_length=V)(mix_input)
    third = Dense(units=nhu, activation='tanh')(mix)
    max_ftr = Lambda(lambda x: np.max(x, axis=0))(third)
    dot_prod = dot([first, max_ftr], axes=2)
    final = add([second, dot_prod])
    activate = Activation('softmax')(final)

    model = Model(inputs=[c_input, ls_input, le_input, gf_input, gv_input, mix_input], outputs=activate)
    model.compile(optimizer='sgd', loss='binary_crossentropy')
    return model


if __name__ == '__main__':
    dataset = "valid"
    if "nt" == os.name:
        path = "E:/Martin/PyCharm Projects/StructuredDataNLG/data/" + dataset
    else:
        path = "/data/matulma4/wikipedia-biography-dataset/wikipedia-biography-dataset/" + dataset
    # embed_table()
    create_model(100, 100)
