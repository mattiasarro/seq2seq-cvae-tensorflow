import tensorflow as tf
import numpy as np
import sys
import unittest
# from copy import deepcopy
sys.path.append("..")

from data_utils import Data, Tokenizer
from hparams import hparams_unit # noqa
import helpers

class TokenizerTest(tf.test.TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer.get_tokenizer(hparams_unit)
        self.special_syms = len(Tokenizer.special) # pad, eos, unk
        self.unk = Tokenizer.special[b"_UNK"]
        self.pad = Tokenizer.special[b"_PAD"]

    def test_tokenizer_setup(self):
        self.assertEqual(self.tokenizer.vocab_size, 13 + self.special_syms)

    def test_encode(self):
        encoded = self.tokenizer.encode("this is a unknown unknown2")
        ints = self.padded([0, 1, 2]) + [self.unk, self.unk]
        self.assertEqual(encoded, ints)

    def test_encode_list_np(self):
        lines = [
            "this is a line",
            "this, on the other hand,"
        ]
        ret = self.tokenizer.encode_list_np(lines)[0]
        self.assertTrue(np.array_equal(self.ints, ret))

    def test_decode(self):
        lines = [
            b"this is a line _PAD",
            b"_UNK on the other _UNK"
        ]
        ret = self.tokenizer.decode(self.ints)
        self.assertEqual(lines, ret)

    @property
    def ints(self):
        ints = [ # words with commas are unk
            self.padded([0, 1, 2, 3]) + [self.pad],
            [self.unk] + self.padded([4, 5, 6]) + [self.unk],
        ]
        return np.array(ints)

    def padded(self, arr):
        return [i + self.special_syms for i in arr]

class DataTest(tf.test.TestCase):

    pos_lines = [
        "this is a line",
        "this, on the other hand, is a longer line",
        "the third line exists as well",
    ]
    neg_lines = [
        "bad movie",
    ]
    unsup_lines = [
        "unsupervised sentence",
        "could be good",
        "could be bad",
        "could be anything in between",
    ]

    def setUp(self):
        self.data = Data("train", hparams_unit)
        self.dataset = self.data.get_dataset(labeled=True)
        self.num_pos = len(self.pos_lines)
        self.num_neg = len(self.neg_lines)
        self.num_unsup = len(self.unsup_lines)

        self.tokenizer = Tokenizer.get_tokenizer(hparams_unit)
        self.all_sup_lines = [
            self.tokenizer.decode(self.tokenizer.encode(line))
            for line in self.pos_lines + self.neg_lines
        ]

    def test_read_all_lines(self):
        with self.test_session() as sess:
            iterator = self.dataset.make_one_shot_iterator().get_next()

            for i in range(self.num_neg + self.num_pos):
                sess.run(iterator)
            with self.assertRaises(tf.errors.OutOfRangeError):
                sess.run(iterator)

    def test_dataset_with_label(self):
        with self.test_session() as sess:
            ds = self.data.dataset_with_label(0, self.data.file_pattern("pos"))
            iterator = ds.make_one_shot_iterator().get_next()

            for i in range(self.num_pos):
                line, label = sess.run(iterator)
                self.assertEqual(0, label)
                self.assertEqual(
                    self.tokenizer.encode(self.pos_lines[i]),
                    self.tokenizer.encode(line),
                )

    def test_batched_dataset(self):
        with self.test_session() as sess:
            dataset = self.data.repeat_and_shuffle(self.dataset)
            iterator = dataset.make_one_shot_iterator().get_next()

            all_seqs = []
            for i in range(3 * (self.num_pos + self.num_neg)): # 3 epochs
                seqs, seq_lens, labels = sess.run(iterator)
                for seq in seqs:
                    all_seqs.append(self.tokenizer.decode(seq))

            for line in self.all_sup_lines:
                self.assertTrue(line in all_seqs)

    def test_dataset_synthetic(self):
        with self.test_session() as sess:
            dataset = self.data.dataset_synthetic()
            iterator = dataset.make_one_shot_iterator().get_next()

            for i in range(10):
                seqs, seq_lens, labels = sess.run(iterator)
                bs = hparams_unit.batch_size

                seqs_trg = np.zeros((bs, hparams_unit.max_seq_len), np.int32)
                seq_lens_trg = np.ones([bs, 1]) * hparams_unit.max_seq_len

                self.assertTrue(np.array_equal(seqs_trg, seqs))
                self.assertTrue(np.array_equal(seq_lens_trg, seq_lens))

    def test_switch_dataset_manual(self):
        inputs, labels = self.data.get_input_fn()()
        hook = self.data.switch_dataset_hook

        with self.test_session() as sess:
            hook.after_create_session(sess, None)

            for task in hparams_unit.train_stage_names:
                hook.switch_to(task, sess)
                inputs_, labels_ = sess.run([inputs, labels])
                self.assertEqual(inputs_["tasks"], task.encode("UTF-8"))

    def test_switch_dataset_routine(self):
        steps_per_stage = 3 # ensure this assumption is met in hparams_unit
        inputs, labels = self.data.get_input_fn()()
        hook = self.data.switch_dataset_hook

        with self.test_session() as sess:
            hook.after_create_session(sess, None)

            for task in hparams_unit.train_stage_names:
                for i in range(steps_per_stage):
                    inputs_, labels_ = sess.run([inputs, labels])

                    fake_context = helpers.Bunch(session=sess)
                    hook.after_run(fake_context, None)

                    self.assertEqual(inputs_["tasks"], task.encode("UTF-8"))

if helpers.run_from_ipython():
    t = DataTest("test_switch_dataset_routine")
    runner = unittest.TextTestRunner()
    runner.run(t)
else:
    tf.test.main()
