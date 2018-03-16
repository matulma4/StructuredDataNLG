from keras import Sequential
from keras.layers import LSTM, Embedding


def create_model():
    model = Sequential()
    model.add(Embedding(voc_size, output_dim, input_len))
    model.add(LSTM(output_dim))
    model.compile('rmsprop', 'mse')
    return model

if __name__ == '__main__':
    # Number of words in a dictionary
    voc_size = 1000

    # Maximum length of a sequence
    input_len = 20

    # Vector dimension
    output_dim = 64