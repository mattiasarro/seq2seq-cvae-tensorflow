import tensorflow as tf
from copy import copy
from hparams import hparams
from data_utils import Tokenizer, get_input_fn

def test_data():

    h = copy(hparams)
    h.dataset = "unit_test"
    (sequences, seq_lens), labels = get_input_fn("train", h)()

    sess = tf.Session()
    s, c, l = sess.run([sequences, seq_lens, labels])

    t = Tokenizer.get_tokenizer(hparams)
    decoded = t.decode(s)[0]
    target = "aside from the terrific sea rescue _UNK of which there are very few i just did not care about any of the _UNK"

    assert decoded == target.split()
    assert l[0][0] == 0

test_data()
