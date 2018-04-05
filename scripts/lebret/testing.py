import inspect
import os
import sys

import pickle

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from config import *


def load_classes():
    path_to_files = path + "pickle/" + dataset
    output = pickle.load(open(path_to_files + "/output.pickle", "rb"))
    infoboxes = pickle.load(open(path_to_files + "/infoboxes.pickle", "rb"))
    return output, infoboxes
