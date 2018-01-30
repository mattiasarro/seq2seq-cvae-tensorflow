from tensorflow.contrib.seq2seq.python.ops import basic_decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest

import tensorflow as tf
import collections

class VAEDecoderOutput(
    collections.namedtuple(
        "VAEDecoderOutput",
        ("rnn_output", "sample_id", "avg_emb"))):
  pass

class VAEDecoder(basic_decoder.BasicDecoder):

    def __init__(self, *args, **kwargs):
        self.temperature = kwargs.pop("temperature")
        self.soft = kwargs.pop("soft")
        self.embedding_size = kwargs.pop("embedding_size")
        super(VAEDecoder, self).__init__(*args, **kwargs)

    @property
    def output_size(self):
        # Return the cell output and the id
        return VAEDecoderOutput(
            rnn_output=self._rnn_output_size(),
            sample_id=tensor_shape.TensorShape([]),
            avg_emb=tensor_shape.TensorShape([self.embedding_size]),
        )

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and int32 (the id)
        dtype = nest.flatten(self._initial_state)[0].dtype
        return VAEDecoderOutput(
            nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            dtypes.int32,
            dtypes.float32,
        )

    def initialize(self, name=None):
        return self._helper.initialize() + (self._initial_state,)

    def step(self, time, inputs, state, name=None):
        with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
            cell_outputs, cell_state = self._cell(inputs, state)
            if self._output_layer is not None:
                logits = self._output_layer(cell_outputs)
                # formula (2) from "Toward Controlled Generation of Text"
                logits = tf.divide(logits, self.temperature)

            sample_ids = self._helper.sample(
                time=time, outputs=logits, state=cell_state
            )
            finished, next_inputs, next_state = self._helper.next_inputs(
                time=time,
                outputs=logits,
                state=cell_state,
                sample_ids=sample_ids
            )

            if self.soft:
                # (batch, vocab) x (vocab, emb) = (batch, emb)
                avg_emb = tf.matmul(logits, self._helper.embedding_matrix)
                next_inputs = self._helper.inputs_with_context(avg_emb)
            else:
                # batch, vocab_ = tf.unstack(tf.shape(logits))
                avg_emb = tf.zeros(shape=(self.batch_size, self.embedding_size), dtype=tf.float32)
            outputs = VAEDecoderOutput(logits, sample_ids, avg_emb)

            return (outputs, next_state, next_inputs, finished)

class VAEDecoderHelper(tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper):

    def __init__(self, inputs, sequence_length, embedding, sampling_probability,
                 time_major=False, seed=None, scheduling_seed=None, name=None,
                 z_sample=None):

        self.embedding_matrix = embedding
        self.context_batch = z_sample

        super(VAEDecoderHelper, self).__init__(
            inputs, sequence_length, embedding, sampling_probability,
            time_major=False, seed=None, scheduling_seed=None, name=None
        )

    def initialize(self, *args, **kwargs):
        finished, next_inputs = super(VAEDecoderHelper, self).initialize(*args, **kwargs)
        next_inputs = self.inputs_with_context(next_inputs)
        return finished, next_inputs

    def inputs_with_context(self, inputs):
        return tf.concat([inputs, self.context_batch], axis=-1)

    def sample(self, time, outputs, state, name=None):
        with ops.name_scope(name, "ScheduledEmbeddingTrainingHelperSample",
                            [time, outputs, state]):

            # Return -1s where we did not sample, and sample_ids elsewhere
            select_sampler = bernoulli.Bernoulli(
                probs=self._sampling_probability, dtype=dtypes.bool)
            select_sample = select_sampler.sample(
                sample_shape=self.batch_size, seed=self._scheduling_seed)
            sample_id_sampler = categorical.Categorical(
                logits=outputs, validate_args=True)
            return array_ops.where(
                select_sample,
                sample_id_sampler.sample(seed=self._seed),
                gen_array_ops.fill([self.batch_size], -1))

    def next_inputs(self, *args, **kwargs):
        finished, next_inputs, state = super(VAEDecoderHelper, self).next_inputs(*args, **kwargs)
        next_inputs = self.inputs_with_context(next_inputs)
        return finished, next_inputs, state
