import inspect
import sys

from config import *

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


class Sample:
    def __init__(self, score, sentence, indexes, t_words, starts, ends, infobox):
        self.sentence = sentence
        self.score = score
        try:
            self.indexes = indexes + [t_words[sentence[-1]]]
        except KeyError:
            self.indexes = indexes + [len(t_words.keys())]
        if sentence[-1] in infobox.keys():
            self.starts = starts + [list(set([j[1] + j[0] * l for j in infobox[sentence[-1]]]))]
            self.ends = ends + [list(set([j[2] + j[0] * l for j in infobox[sentence[-1]]]))]
        else:
            self.starts = starts + [[]]
            self.ends = ends + [[]]
