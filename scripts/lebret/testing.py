import inspect
import os
import sys

import pickle

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from config import *


def load_from_file():
    path_to_files = path + "pickle/" + dataset
    output = pickle.load(open(path_to_files + "/output.pickle", "rb"))
    t_fields = pickle.load(open(path_to_files + "/t_fields.pickle", "rb"))
    t_words = pickle.load(open(path_to_files + "/t_words.pickle", "rb"))
    infoboxes = pickle.load(open(path_to_files + "/infoboxes.pickle", "rb"))
    return t_fields, t_words, infoboxes, output


if __name__ == '__main__':
    pass
