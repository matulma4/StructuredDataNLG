import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import sys

from config import *
from sample import Sample
from testing import make_sample, global_conditioning, process_infoboxes, load_from_file


def make_heatmap(h_data):
    # uniform_data = np.ones((10, 12))
    ax = sns.heatmap(h_data, linewidth=0.5)
    plt.show()


def get_attention(model, sentence, word_tf, field_tf, infobox, loc_dim, t_f, t_w):
    global l, V
    l = int(m_name[8:10])
    h = m_name[:-6]
    infoboxes, output, model, field_transform, word_transform = load_from_file("test_r", m_name, h)
    with open(path + "pickle/" + dataset + "/" + h + "/params.txt") as f:
        V, max_loc_idx, glob_field_dim, glob_word_dim, loc_dim, f_len, w_len, w_count = [int(a) for a in
                                                                                         f.read().split()]

    sentences = [
        ['the', 'address', 'of', 'sotto', 'is', 'address_1', 'at', 'address_1', 'at', 'address_1', 'at', 'address_1',
         'at', 'address_1', 'at', 'address_1', 'at', 'address_1', 'at', 'address_1'],
        ['there', 'are', '<UNK>', 'restaurant', '-s', 'in', 'the', 'square', 'luther', 'area', 'that', 'serve',
         'comfort', 'food', 'and', 'does', 'not', 'allow', 'child', '-s']]

    t_fields, t_words, ib = process_infoboxes(infoboxes, field_transform, word_transform)
    for i in range(2):
        r = get_attention(model, sentences[i], word_transform, field_transform, ib[i], loc_dim, t_fields[i], t_words[i])
        make_heatmap(np.array(r))
    pass
    s = Sample(0.0, ["s" + str(i) for i in range(l)], [word_tf["s" + str(i)] for i in range(l - 1)], word_tf, field_tf,
               [[len(field_tf) * l + 2] for _ in range(l - 1)], [[len(field_tf) * l + 2] for _ in range(l - 1)],
               infobox)
    mix_sample = []
    result = []
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
        result.append(prediction[0][V:])
        s = Sample(0.0, s.indexes + [word], s.indexes, word_tf, field_tf,
               s.starts, s.ends,
               infobox)
    # return attentions
    return result

if __name__ == '__main__':
    m_name = sys.argv[1]
