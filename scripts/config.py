import os

limit = 10000
l = 10
d = 64
g = 128
n = 11
V = 20000
nhu = 256

dataset = "valid"
if "nt" == os.name:
    path = "E:/Martin/PyCharm Projects/StructuredDataNLG/data/" + dataset
else:
    path = "/data/matulma4/wikipedia-biography-dataset/wikipedia-biography-dataset/" + dataset
