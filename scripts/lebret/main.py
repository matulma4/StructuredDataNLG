import os

from keras.layers import Embedding, Input, Dense, Activation, concatenate, maximum, dot, add
from keras.models import Sequential, Model

from data_loader import load_infoboxes, get_most_frequent

d = 64
g = 128
l = 10
n = 11
V = 20000
nhu = 100


def embed_table():
    r, f = load_infoboxes(path, dataset)
    # TODO Get fields occurring at least 100 times
    result = get_most_frequent(r, False)

    f = len("")

    pass


def create_model(loc_dim, glob_dim):
    model = Sequential()

    c_input = Input(shape=(l,))
    ls_input = Input(shape=(l,))
    le_input = Input(shape=(l,))
    gf_input = Input(shape=(l,))
    gv_input = Input(shape=(l,))

    context = Embedding(input_dim=V, output_dim=d, input_length=l)(c_input)
    local_start = Embedding(input_dim=loc_dim, output_dim=d, input_length=l)(ls_input)
    local_end = Embedding(input_dim=loc_dim, output_dim=d, input_length=l)(le_input)
    global_field = Embedding(input_dim=glob_dim, output_dim=g, input_length=l)(gf_input)
    global_value = Embedding(input_dim=glob_dim, output_dim=g, input_length=l)(gv_input)

    merged = concatenate([context, local_start, local_end, global_field, global_value])

    first = Dense(units=nhu, activation='tanh')(merged) # h(x)
    second = Dense(units=V)(first)

    third = Dense(units=nhu, activation='tanh')(local_start)
    max_ftr = maximum([third, third])
    dot_prod = dot(first, max_ftr)
    final = add([second, dot_prod])
    activate = Activation('softmax')(final)

    model = Model(inputs=[c_input, ls_input, le_input, gf_input, gv_input], outputs=activate)
    model.compile(optimizer='sgd', loss='binary_crossentropy')


if __name__ == '__main__':
    dataset = "valid"
    if "nt" == os.name:
        path = "E:/Martin/PyCharm Projects/StructuredDataNLG/data/" + dataset
    else:
        path = "/data/matulma4/wikipedia-biography-dataset/wikipedia-biography-dataset/" + dataset
    embed_table()
