class Sample:
    def __init__(self, score, sentence, indexes, t_words):
        self.sentence = sentence
        self.score = score
        try:
            self.indexes = indexes + [t_words[sentence[-1]]]
        except KeyError:
            self.indexes = indexes + [len(t_words.keys())]
