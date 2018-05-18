import sys
import inspect
import os
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from config import *


def make_heatmap(h_data):
    # uniform_data = np.ones((10, 12))
    ax = sns.heatmap(h_data, linewidth=0.5)
    plt.show()


def get_attention(m_name):
    model_name = sys.argv[1]
    from sample import Sample
    from testing import make_sample, global_conditioning, process_infoboxes, load_from_file
    global l, V
    l = int(m_name[8:10])
    h = m_name[:-6]
    infoboxes, output, model, field_tf, word_tf = load_from_file("test_r", m_name, h)
    with open(path + "pickle/" + dataset + "/" + h + "/params.txt") as f:
        V, max_loc_idx, glob_field_dim, glob_word_dim, loc_dim, f_len, w_len, w_count = [int(a) for a in
                                                                                         f.read().split()]

    sentences = [
        ['the', 'address', 'of', 'sotto', 'is', 'address_1', 'at', 'address_1', 'at', 'address_1', 'at', 'address_1',
         'at', 'address_1', 'at', 'address_1', 'at', 'address_1', 'at', 'address_1'],
        ['there', 'are', '<UNK>', 'restaurant', '-s', 'in', 'the', 'square', 'luther', 'area', 'that', 'serve',
         'comfort', 'food', 'and', 'does', 'not', 'allow', 'child', '-s']]

    t_fields, t_words, ib = process_infoboxes(infoboxes, field_tf, word_tf)
    for k in range(2):
        t_f = t_fields[k]
        t_w = t_words[k]
        infobox = ib[k]
        sentence = sentences[k]
        #     r = get_attention(model, sentences[k], word_transform, field_transform, ib[k], loc_dim, t_fields[k], t_words[k])
        #     make_heatmap(np.array(r))
        #     pass
        s = Sample(0.0, ["s" + str(i) for i in range(l)], [word_tf["s" + str(i)] for i in range(l - 1)], word_tf,
                   field_tf,
                   [[len(field_tf) * l + 2] for _ in range(l - 1)], [[len(field_tf) * l + 2] for _ in range(l - 1)],
                   infobox)
        mix_sample = []
        vector = []
        for t_key in infobox:
            vt = np.unique([tv[0] * l + tv[1] for tv in infobox[t_key]])
            mix_sample.append(np.pad(vt, (0, loc_dim - vt.shape[0]), mode='constant'))
        mix_sample = np.pad(np.array(mix_sample), ((0, w_count - len(mix_sample)), (0, 0)), mode='constant')
        gf, gw = global_conditioning(t_f, t_w, f_len, w_len)
        for j in range(l, len(sentence)):
            word = sentence[j]
            # construct sample
            samples_context, samples_ls, samples_le = make_sample(s, loc_dim)
            # predict
            prediction = model.predict(
                {'c_input': np.array(samples_context), 'ls_input': np.array(samples_ls),
                 'le_input': np.array(samples_le),
                 'gf_input': np.array(gf),
                 'gw_input': np.array(gw),
                 'mix_input': np.array([mix_sample])
                 })
            # save part of vector
            vector.append(prediction[0][V:])
            s = Sample(0.0, s.indexes + [word], s.indexes, word_tf, field_tf,
                       s.starts, s.ends,
                       infobox)
        make_heatmap(np.array(vector))
        # return attentions
        # return vector


def make_bar(data, labels, h, baseline):
    colors = ['g']*2 +['b'] + ['y']*2  + ['c']*3
    fig, ax = plt.subplots()
    index = np.arange(1, len(data) + 1)
    width = 0.35 
    bar = 0
    for i in range(len(data)):
        plt.bar(i - width/2, data[i], width,
                color=colors[i],
                label=labels[i])
        # bar += width
    plt.xlabel('Group')
    plt.ylabel('BLEU')
    plt.title('Results')
    plt.xticks(index-1 + (bar - width) / 2, ('abcd'))
    plt.axhline(baseline)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(path + "pictures/" + h + ".png", dpi=300)


if __name__ == '__main__':
    # global/local | remove punc | n | beam 
    d = [0.2645257132032,0.26864211575348507,0.28183018508296925,0.2726072061388273,0.2787024084272768,0.2779071111844879,0.2782477751992007, 0.2779647131390378]
    lab = ['Glob/loc','Local','Remove punc','5-gram','15-gram','5','15','20']
    make_bar(d, lab, 'rest_bleu', 0.2776786961450034)
