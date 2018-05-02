from flask import Flask
from testing import beam_search, load_from_file, process_infoboxes, global_conditioning
from config import *
import numpy as np
app = Flask(__name__)


@app.route("/")
def default():
    return "Hello World!"


@app.route("/hello/<int:index>/")
def hello(index):
    infobox = ib[index]
    gf, gw = global_conditioning(t_fields[index], t_words[index], f_len, w_len)
    mix_sample = []
    for t_key in infobox:
        vt = np.unique([tv[0] * l + tv[1] for tv in infobox[t_key]])
        mix_sample.append(np.pad(vt, (0, loc_dim - vt.shape[0]), mode='constant'))
    return " ".join(beam_search(model, 10, l+30, output, word_transform, field_transform, gf, gw, infobox, mix_sample, loc_dim, False)[l:])


if __name__ == "__main__":
    m_name = "20180430110201000010"
    h = m_name[:-3]
    infoboxes, output, model, field_transform, word_transform = load_from_file("valid", m_name, h)
    with open(path + "pickle/" + dataset + "/" + h + "/params.txt") as f:
        V, max_loc_idx, glob_field_dim, glob_word_dim, loc_dim, f_len, w_len, w_count = [int(a) for a in
                                                                                         f.read().split()]
    global l
    l = int(m_name[8:10])

    t_fields, t_words, ib = process_infoboxes(infoboxes, field_transform, word_transform)
    app.run()