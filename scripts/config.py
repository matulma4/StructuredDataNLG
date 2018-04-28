import os

l = 10
V = 10000

# data_process
min_keys = 1
limit = 10

# main
sample_limit = 10
d = 64
g = 128
nhu = 256
n_iter = 100
decay_rate = 0


global_cond = True
local_cond = False
use_ft = True
use_mix = False
drop_punc = False


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

