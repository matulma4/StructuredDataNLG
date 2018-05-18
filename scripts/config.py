import os

# Main parameters
l = 10
V = 10000

# Parameters for data_process
min_keys = 1
limit = 10
drop_punc = False

# Parameters for training
sample_limit = 10
d = 64
g = 128
nhu = 256
n_iter = 20
decay_rate = 0

global_cond = True
local_cond = False
use_ft = True
use_mix = False


# Path to files on Windows and Linux
dataset = "valid"
if "nt" == os.name:
    path = "E:/Martin/PyCharm Projects/StructuredDataNLG/"
    data_path = path + "data/" + dataset
    test_path = path + "data/"
    use_ft = False
else:
    path = "/data/matulma4/diploma_thesis/StructuredDataNLG/"
    test_path = "/data/matulma4/diploma_thesis/wikipedia-biography-dataset/wikipedia-biography-dataset/"
    data_path = test_path + dataset
