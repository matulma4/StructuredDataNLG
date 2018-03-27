import os
limit = 10000
l = 10
dataset = "valid"
if "nt" == os.name:
    path = "E:/Martin/PyCharm Projects/StructuredDataNLG/data/" + dataset
else:
    path = "/data/matulma4/wikipedia-biography-dataset/wikipedia-biography-dataset/" + dataset