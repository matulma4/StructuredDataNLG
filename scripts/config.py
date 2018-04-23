import os

limit = 1
sample_limit = 1000
l = 10
d = 64
g = 128
n = 11
V = 20000
nhu = 256
min_keys = 1
n_iter = 100
decay_rate = 0

global_cond = True
local_cond = True
use_ft = True
use_mix = False


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

