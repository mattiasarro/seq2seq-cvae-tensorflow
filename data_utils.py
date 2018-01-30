import tensorflow as tf
import numpy as np
from tensorflow.contrib.data import Dataset, TextLineDataset
from helpers import pr, lazy # noqa
from copy import copy

class Tokenizer(object):

    tokenizers = {}
    special = {b"_PAD": 0, b"_EOS": 1, b"_UNK": 2}

    @classmethod
    def get_tokenizer(cls, hparams):
        if hparams.dataset in Tokenizer.tokenizers:
            return Tokenizer.tokenizers[hparams.dataset]
        else:
            vocab_fn = '%s/%s.vocab' % (hparams.data_dir, hparams.dataset)
            tokenizer = Tokenizer(vocab_fn)
            Tokenizer.tokenizers[hparams.dataset] = tokenizer
            return tokenizer

    def __init__(self, fn):
        self.vocab = self.load_vocab(fn)
        self.vocab_size = len(self.vocab)
        self.i_to_word = {i: word for word, i in self.vocab.items()}

    def load_vocab(self, fn):
        vocab = copy(Tokenizer.special)
        offset = len(vocab)

        with open(fn) as f:
            for i, word in enumerate(f.readlines()):
                vocab[word.strip().encode('UTF-8')] = offset + i

        return vocab

    def encode(self, text):
        if isinstance(text, str):
            text = text.encode('UTF-8')

        tokens = text.split()
        ints = [
            self.vocab.get(word.lower(), self.vocab[b"_UNK"])
            for word in tokens
        ]
        return ints

    def encode_np(self, text):
        return np.array(self.encode(text), dtype=np.int32)

    def encode_list_np(self, lines):
        sequences = [self.encode_np(line) for line in lines]

        seq_lens = [s.shape[0] for s in sequences]
        seq_lens_np = np.array(seq_lens, dtype=np.int32)
        seq_lens_np = np.expand_dims(seq_lens_np, 1)

        zeros = np.zeros([len(sequences), max(seq_lens)], dtype=np.int32)
        for i, s in enumerate(sequences):
            zeros[i, :s.shape[0]] = s

        return zeros, seq_lens_np

    def decode(self, ints):
        to_word = lambda i: self.i_to_word.get(i, "UNK")
        is_np_vector = type(ints) == np.ndarray and len(ints.shape) == 1

        if type(ints) == list or is_np_vector:
            words = [to_word(i) for i in ints]
            return b" ".join(words)

        elif type(ints) == np.ndarray:
            if len(ints.shape) == 2:
                sentences = []
                batch, seq_len = ints.shape
                for b_i in range(batch):
                    words = [
                        to_word(ints[b_i, t_i])
                        for t_i in range(seq_len)
                    ]
                    sentences.append(b" ".join(words))

                return sentences
            else: raise Exception

class Data(object):

    def __init__(self, eval_train, hparams):
        self.hparams = hparams
        self.eval_train = eval_train
        self.input_fn = self.get_input_fn() # ensure it's called in order

    def get_input_fn_for(self, model_class):
        def task(task_name):
            s = tf.constant(task_name, tf.string)
            return Dataset.from_tensors(s).repeat()

        def labeled():
            lab_unshuf = self.get_dataset(labeled=True)
            return self.repeat_and_shuffle(lab_unshuf)

        def input_fn():
            name = model_class.__name__
            if name == "ClassifierModel":
                dataset = Dataset.zip((labeled(), task("classify")))
            elif name == "ClassifierSynthModel":
                synthetic = self.dataset_synthetic()
                dataset = Dataset.zip((synthetic, task("classify_synth")))
            elif name == "VAECondModel":
                dataset = Dataset.zip((labeled(), task("vae_cond_lab")))

            iterator = dataset.make_one_shot_iterator()
            (sequences, seq_lens, labels), tasks = iterator.get_next()
            inputs = {
                "sequences": sequences,
                "seq_lens": seq_lens,
                "tasks": tasks
            }
            return inputs, labels
        return input_fn

    def get_input_fn(self):
        self.switch_dataset_hook = SwitchDatasetHook(
            self.hparams, self.eval_train)

        def input_fn():
            with tf.variable_scope("Dataset"):
                datasets = self.datasets()
                iterator = tf.contrib.data.Iterator.from_structure(
                    datasets["classify"].output_types,
                    datasets["classify"].output_shapes
                )
                self.switch_dataset_hook.switch_to_ops = {
                    name: iterator.make_initializer(dataset)
                    for name, dataset in datasets.items()
                }

                (sequences, seq_lens, labels), tasks = iterator.get_next()
                inputs = {
                    "sequences": sequences,
                    "seq_lens": seq_lens,
                    "tasks": tasks
                }
                return inputs, labels
        return input_fn

    def datasets(self):
        synthetic = self.dataset_synthetic()
        lab_unshuf = self.get_dataset(labeled=True)
        unlab_unshuf = self.get_dataset(labeled=False)
        labeled = self.repeat_and_shuffle(lab_unshuf)
        unlabeled = self.repeat_and_shuffle(unlab_unshuf)
        lab_unlab = self.repeat_and_shuffle(
            lab_unshuf.concatenate(unlab_unshuf))

        def task(task_name):
            s = tf.constant(task_name, tf.string)
            return Dataset.from_tensors(s).repeat()

        return {
            "vae_uncond": Dataset.zip((lab_unlab, task("vae_uncond"))),
            "classify": Dataset.zip((labeled, task("classify"))),
            "vae_cond_lab": Dataset.zip((labeled, task("vae_cond_lab"))),
            "vae_cond_unlab": Dataset.zip((unlabeled, task("vae_cond_unlab"))),
            "classify_synth": Dataset.zip((synthetic, task("classify_synth"))),
        }

    def get_dataset(self, labeled=True):
        if labeled:
            neg = self.dataset_with_label(0, self.file_pattern("neg"))
            pos = self.dataset_with_label(1, self.file_pattern("pos"))
            line_label = neg.concatenate(pos)
        else:
            line_label = self.dataset_with_label(-1, self.file_pattern("unsup"))
        dataset = line_label.map(self.tokenize)
        return dataset

    def tokenize(self, line, label):
        def _tokenize(line, label):
            sequence = Tokenizer.get_tokenizer(self.hparams).encode_np(line)
            count = np.array(sequence.shape[0], dtype=np.int32)
            label = np.array(label, dtype=np.int32)

            # all outputs must have the same nr of axes
            count = np.expand_dims(count, axis=1)
            label = np.expand_dims(label, axis=1)

            return sequence, count, label

        return tuple(tf.py_func(_tokenize,
                                [line, label],
                                [tf.int32, tf.int32, tf.int32],
                                ))

    def repeat_and_shuffle(self, dataset):
        return dataset.repeat() \
            .shuffle(buffer_size=self.hparams.buffer_size) \
            .padded_batch(
                self.hparams.batch_size,
                padded_shapes=([None], [None], [None]))

    def dataset_with_label(self, label_int, src_pattern):
        label = tf.constant(label_int, tf.int32, name="label")
        lines = Dataset.list_files(src_pattern).flat_map(
            lambda fn: TextLineDataset(fn)
        )
        labels = Dataset.from_tensors(label).repeat()
        return Dataset.zip((lines, labels))

    def dataset_synthetic(self):
        # this is not actually generating a synthetic dataset but creates
        # dummy dataset that will result in the generation of synthetic data

        nc = self.hparams.num_classes
        seqs = tf.zeros(shape=[nc, self.hparams.max_seq_len], dtype=tf.int32)
        seq_lens = tf.ones(shape=[nc, 1], dtype=tf.int32)
        seq_lens *= self.hparams.max_seq_len
        labels = tf.constant(np.arange(nc), dtype=tf.int32)
        labels = tf.reshape(labels, [nc, 1])

        seqs = Dataset.from_tensor_slices(seqs)
        seq_lens = Dataset.from_tensor_slices(seq_lens)
        labels = Dataset.from_tensor_slices(labels)
        dataset = Dataset.zip((seqs, seq_lens, labels))
        dataset = self.repeat_and_shuffle(dataset)

        return dataset

    def file_pattern(self, pos_neg):
        fn = "/%s/%s/*.txt" % (self.eval_train, pos_neg)
        return self.hparams.data_dir + fn

class SwitchDatasetHook(tf.train.SessionRunHook):

    dataset = None
    dataset_steps_done = 0

    def __init__(self, hparams, eval_train):
        self.eval_train = eval_train
        self.train_stages = hparams.train_stages
        SwitchDatasetHook.dataset = self.train_stages[0][0]

    def after_create_session(self, session, coord):
        self.switch_to(SwitchDatasetHook.dataset, session)

    def after_run(self, context, vals):
        if self.eval_train == "eval":
            return

        if self.all_dataset_steps_done:
            self.switch_to(self.next_dataset, context.session)
            SwitchDatasetHook.dataset = self.next_dataset
            SwitchDatasetHook.dataset_steps_done = 0
        else:
            SwitchDatasetHook.dataset_steps_done += 1

    def switch_to(self, dataset, session):
        session.run(self.switch_to_ops[dataset])
        if dataset != SwitchDatasetHook.dataset:
            print("[%s] Switched to dataset %s" % (self.eval_train, dataset))
        else:
            print("[%s] Using dataset %s" % (self.eval_train, dataset))

    @property
    def all_dataset_steps_done(self):
        return SwitchDatasetHook.dataset_steps_done == self.num_dataset_steps

    @property
    def num_dataset_steps(self):
        for name, cnt in self.train_stages:
            if name == SwitchDatasetHook.dataset:
                return cnt

    @property
    def next_dataset(self):
        try:
            for i, tup in enumerate(self.train_stages):
                dataset, cnt = tup
                if dataset == SwitchDatasetHook.dataset:
                    return self.train_stages[i + 1][0]
        except IndexError:
            return self.train_stages[0][0]
