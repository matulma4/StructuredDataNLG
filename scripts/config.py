import os

limit = 500
sample_limit = 1000
l = 10
d = 64
g = 128
n = 11
V = 20000
nhu = 256

dataset = "valid"
if "nt" == os.name:
    path = "E:/Martin/PyCharm Projects/StructuredDataNLG/"
    data_path = path + "data/" + dataset
else:
    path = "/data/matulma4/diploma_thesis/StructuredDataNLG/"
    data_path = "/data/matulma4/diploma_thesis/wikipedia-biography-dataset/wikipedia-biography-dataset/" + dataset

