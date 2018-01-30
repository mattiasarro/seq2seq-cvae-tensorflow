# %%
import math
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.layers.core import Dense
from decoder_helpers import VAEDecoder, VAEDecoderHelper

from helpers import lazy, pr, get_loss_op # noqa

class ModelComponent(object):

    @classmethod
    def define(cls, *args, **kwargs):
        return cls.get_model(False, *args, **kwargs)

    @classmethod
    def reuse_params(cls, *args, **kwargs):
        return cls.get_model(True, *args, **kwargs)

    @classmethod
    def get_model(cls, reuse, *args, **kwargs):
        with tf.variable_scope(cls.__name__, reuse=reuse) as scope:
            component = cls(*args, **kwargs)
            component.scope = scope
            return component

class Encoder(ModelComponent):

    def __init__(self, cell, model, input_embs=None, use_z=True):
        self.hparams = model.hparams
        self.src_seq_len = model.seq_lens
        self.tasks = model.tasks
        self.global_step = model.global_step
        if input_embs is None:
            self.src_seq_emb = tf.nn.embedding_lookup(
                model.embedding_matrix, model.sequences)
        else:
            self.src_seq_emb = input_embs

        self._init_bidirectional(cell)
        if use_z:
            self._init_z()

    def _init_bidirectional(self, cell):
        with tf.variable_scope("bidirectional"):

            inputs = tf.Print(self.src_seq_emb, [self.tasks, self.global_step], "-- src_seq_emb passing thru enc bi --")
            ((fw_outputs, bw_outputs), (fw_state, bw_state)) = (
                tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell,
                    cell_bw=cell,
                    inputs=inputs,
                    sequence_length=self.src_seq_len,
                    dtype=tf.float32)
            )

            # outputs = tf.concat((fw_outputs, bw_outputs), 2)
            state_c = tf.concat((fw_state.c, bw_state.c), axis=1)
            state_h = tf.concat((fw_state.h, bw_state.h), axis=1)
            self.state = LSTMStateTuple(c=state_c, h=state_h)

    def _init_z(self):
        with tf.variable_scope("z"):
            if not self.hparams.variational:
                self.z_sample = self.state
                return

            h = self.state.h
            self.z_mu = Dense(self.hparams.z_units, name="z_mu")(h)
            self.z_log_var = Dense(self.hparams.z_units, name="z_log_var")(h)
            self.z_var = tf.exp(self.z_log_var, name="z_var")

            z = tf.distributions.Normal(
                loc=self.z_mu, # mean
                scale=tf.sqrt(self.z_var), # std dev
                allow_nan_stats=False,
                name="z_distribution")

            self.z_sample = z.sample(name="z_sample")

class Decoder(ModelComponent):

    PAD = 0
    EOS = 1

    def __init__(self, cell, model, z_sample, encoder=None,
                 soft=False, sampling_prob=None):
        self.cell = cell
        self.encoder = encoder
        self.z_sample = z_sample
        self.trg_seq = model.sequences
        self.trg_seq_len = model.seq_lens
        self.embedding_matrix = model.embedding_matrix
        self.vocab_size = model.vocab_size
        self.global_step = model.global_step
        self.hparams = model.hparams
        self.soft = soft
        self.fixed_sampling_prob = sampling_prob

        if model.mode in ["train", "eval"]:
            self._init_train_inputs_and_targets()
            self._init_train()
        else:
            self._init_train_inputs_and_targets() # TODO can we avoid this?
            self._init_inference()

    def _init_train_inputs_and_targets(self):
        with tf.variable_scope('Inputs'):
            batch_size, sequence_size = tf.unstack(tf.shape(self.trg_seq))

            # TODO separate symbols for EOS and SOS
            self.EOS_SLICE = self.EOS * tf.ones([batch_size, 1], dtype=tf.int32, name="EOS_slice")
            self.PAD_SLICE = self.PAD * tf.ones([batch_size, 1], dtype=tf.int32, name="PAD_slice")

            train_inputs = tf.concat([self.EOS_SLICE, self.trg_seq], axis=1)
            self.train_len = self.trg_seq_len + 1
            self.train_inputs_emb = tf.nn.embedding_lookup(
                self.embedding_matrix, train_inputs, name="train_inputs_emb")

            # put EOS symbol at the end of target sequence
            train_targets = tf.concat([self.trg_seq, self.PAD_SLICE], axis=1)
            train_targets_eos_mask = tf.one_hot( # (batch, t) = (len(trg_seq_len), sequence_size + 1)
                self.trg_seq_len,
                sequence_size + 1,
                on_value=self.EOS,
                off_value=self.PAD,
                dtype=tf.int32)
            self.train_targets = tf.add(train_targets, train_targets_eos_mask, name="train_targets")

            self.loss_weights = tf.ones([
                batch_size,
                tf.reduce_max(self.train_len),
            ], dtype=tf.float32, name="loss_weights")

    def _init_train(self):
        helper = VAEDecoderHelper(
            self.train_inputs_emb,
            self.train_len,
            self.embedding_matrix,
            z_sample=self.z_sample,
            sampling_probability=self.sampling_prob,
            name="VAEDecoderHelper",
        )

        outputs = self._decode(helper)
        self.logits = outputs.rnn_output
        self.predictions = outputs.sample_id
        self.avg_emb = outputs.avg_emb

    def _init_inference(self):
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper( # TODO z_sample
            self.embedding_matrix,
            tf.squeeze(self.EOS_SLICE, axis=[1]),
            self.EOS)

        outputs = self._decode(helper)
        self.predictions = outputs.sample_id

    def _decode(self, helper):
        decoder = VAEDecoder(
            self.cell,
            helper,
            self.initial_state(),
            output_layer=self.output_network,
            temperature=self.sampling_temperature,
            soft=self.soft,
            embedding_size=self.hparams.embedding_size,
        )

        outputs, state, seq_len = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            swap_memory=True,
            maximum_iterations=self.hparams.max_seq_len
        )
        return outputs

    @lazy
    def output_network(self):
        # TODO why not use bias?
        return Dense(self.vocab_size, use_bias=False, name="decoder_out")

    @lazy
    def sampling_prob(self):
        if self.fixed_sampling_prob is not None:
            return self.fixed_sampling_prob

        # inverse sigmoid decay for scheduled sampling https://arxiv.org/abs/1506.03099
        k = tf.constant(self.hparams.scheduled_sampling_k, dtype=tf.float32)
        i = tf.cast(self.global_step, tf.float32)
        p = k / (k + tf.exp(i / k))
        return p

    @lazy
    def sampling_temperature(self):
        # formula 2 of "Toward Controlled Generation of Text"
        # TODO anneal from 1 to 0
        tau = tf.Variable(1.0, name="sampling_temperature", trainable=False)
        return tau

    @property
    def recon_loss(self):
        return seq2seq.sequence_loss(
            logits=self.logits,
            targets=self.train_targets, # * max sequence length?
            weights=self.loss_weights,
            name="reconstruction_loss",
        )

    def initial_state(self):
        if self.encoder is None:
            batch_shape = [self.hparams.batch_size, self.hparams.dec_units]
            return LSTMStateTuple(
                c=tf.zeros(batch_shape, dtype=tf.float32),
                h=tf.zeros(batch_shape, dtype=tf.float32),
            )
        else:
            return LSTMStateTuple(
                c=tf.zeros_like(self.encoder.state.c),
                h=tf.zeros_like(self.encoder.state.h),
            )

class Classifier(ModelComponent):

    def __init__(self, cell, model, input_embs=None, encoder=None):
        self.global_step = model.global_step
        self.hparams = model.hparams
        self.labels = model.labels

        if encoder: # used by soft (and synth if has_stage(vae_cond_lab))
            self.encoder = encoder
        else:
            with tf.variable_scope("Encoder"):
                self.encoder = Encoder(
                    cell=cell,
                    model=model,
                    input_embs=input_embs,
                    use_z=False,
                )

        self.loss = self.get_loss()

    @lazy
    def logits(self):
        out_net = Dense(self.hparams.num_classes, use_bias=False, name="logits")
        return out_net(self.encoder.state.h)

    @lazy
    def predictions(self):
        return tf.cast(
            tf.argmax(self.logits, axis=1),
            tf.int32,
            name="predictions"
        )

    def get_loss(self):
        labels = tf.one_hot(self.labels, self.hparams.num_classes)
        loss = tf.losses.softmax_cross_entropy(
            logits=self.logits,
            onehot_labels=labels,
        )
        loss = tf.reduce_sum(loss, name="loss_sum")
        return loss

class Model(object):

    all_comps = ["Encoder", "Decoder", "Classifier"]

    def __init__(self, mode, features, labels, vocab_size, hparams):
        print("[%s] Initializing model" % mode)
        self.mode = mode
        self.vocab_size = vocab_size
        self.hparams = hparams
        self.sequences = features["sequences"]
        self.seq_lens = tf.squeeze(features["seq_lens"], axis=1) # undo the expand_dims in input_fn,
        self.labels = tf.squeeze(labels, axis=1) if labels is not None else None
        self.embedding_matrix = self._get_embeddings()
        self.tasks = features["tasks"]
        if "tasks" in hparams.log:
            self.tasks = tf.Print(self.tasks, [self.tasks], "-- (%s) TASK " % mode)
        self.init_model()

    def _get_embeddings(self):
        with tf.variable_scope("embedding"):

            sqrt3 = math.sqrt(3) # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            return tf.get_variable(
                name="embedding_matrix",
                shape=[self.vocab_size, self.hparams.embedding_size],
                initializer=initializer,
                dtype=tf.float32)

    @lazy
    def train_op(self):
        if self.mode in ["train", "eval"]:
            with tf.variable_scope("train_op", reuse=False):
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.hparams.learning_rate,
                )

                train_ops = [
                    self.minimize_component_loss(comp)
                    for comp in self.comps
                ]
                self.setup_unused_optimizer_params()

                inc_step = tf.assign_add(self.global_step, 1)
                with tf.control_dependencies([inc_step]):
                    train_op = tf.group(*train_ops, name="train_op_group")

                return train_op

    def minimize_component_loss(self, comp):
        return self.optimizer.minimize(
            loss=self.loss, # TODO ensure it's not calculated many times
            var_list=self.vars_in_scopes([comp, "embedding"]),
            name="Minimize" + comp,
        )

    def setup_unused_optimizer_params(self):
        # need to declare optimizer ops that are used by non-chief workers,
        # otherwise the chief worker won't initialize those optimizer vars

        if self.is_chief:
            for comp in Model.all_comps:
                if comp not in self.comps:
                    print("successful setup_unused_optimizer_params", comp)
                    self.minimize_component_loss(comp)
                else:
                    print(comp, "in", self.comps)
        else:
            print("-- not chief --")

    @lazy
    def is_chief(self):
        return type(self).__name__ == self.hparams.worker_processes[0]

    @lazy
    def global_step(self):
        return (tf.contrib.framework.get_global_step() or
                tf.Variable(0, trainable=False, name='global_step'))

    @property
    def eval_metrics(self):
        return {}

    def to_metrics(self, loss_components):
        return {
            k: tf.contrib.metrics.streaming_mean(v)
            for k, v in loss_components.items()
        }

    def vars_in_scopes(self, scopes):
        vars = []
        for scope in scopes:
            v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            vars.extend(v)
        return vars

class VAEModel(Model):

    @property
    def comps(self):
        return ["Encoder", "Decoder"]

    def init_model(self):
        self.encoder = Encoder.define( # tb Encoder
            cell=LSTMCell(self.hparams.enc_units),
            model=self,
        )

        # identity needed so that z is not re-sampled?
        self.z_sample = tf.identity(self.encoder.z_sample, name="z_sample_copy")
        self.decoder = Decoder.define( # tb Decoder
            cell=LSTMCell(self.hparams.dec_units),
            model=self,
            encoder=self.encoder,
            z_sample=self.z_sample,
        )

    @get_loss_op
    def vae_loss(self):
        return self.recon_loss + self.kl_loss

    @get_loss_op
    def recon_loss(self):
        return self.decoder.recon_loss

    @get_loss_op
    def kl_loss(self):
        mu = self.encoder.z_mu
        var = self.encoder.z_var
        log_var = self.encoder.z_log_var

        kl = 1 + log_var - tf.square(mu) - var
        kl = -0.5 * tf.reduce_sum(kl, axis=-1)
        kl = tf.reduce_mean(kl, axis=-1) # average the KL loss of all the training samples

        return self.kl_weight * kl

    @get_loss_op
    def kl_weight(self):
        # see ad_hoc/sigmoid.py for example schedules
        multiplier = tf.constant(self.hparams.kl_weight_multiplier, dtype=tf.float32)
        offset = tf.constant(self.hparams.kl_weight_offset, dtype=tf.float32)
        x = tf.cast(self.global_step, tf.float32)
        return 1 / (1 + tf.exp((multiplier * -x) + offset))

    @property
    def perplexity(self):
        return tf.exp(self.recon_loss)

    @property
    def eval_metrics(self):
        return self.to_metrics({
            'perplexity': self.perplexity,
            'KL': self.kl_loss,
            'recon': self.recon_loss,
        })

class VAEUncondModel(VAEModel):

    @lazy
    def loss(self):
        return self.vae_loss

    @lazy
    def predictions(self):
        return self.decoder.predictions

class VAECondModel(VAEModel):

    def init_model(self):
        super(VAECondModel, self).init_model()

        self.decoder_soft = Decoder.reuse_params( # tb Decoder_1
            cell=LSTMCell(self.hparams.dec_units),
            model=self,
            encoder=self.encoder,
            z_sample=self.z_sample,
            soft=True,
        )
        # TODO test that encoder_soft only encodes once,
        # not once for disent_loss and once for attr_preserve_loss
        self.encoder_soft = Encoder.reuse_params( # tb Encoder_1
            cell=LSTMCell(self.hparams.enc_units),
            model=self,
            input_embs=self.decoder_soft.avg_emb,
        )
        self.classifier_soft = Classifier.define( # tb Classifier_1
            cell=LSTMCell(self.hparams.enc_units),
            model=self,
            input_embs=self.decoder_soft.avg_emb,
            # encoder=self.encoder_soft, # can't use because VAECondModel is chief and needs to define classifier ops
        )

    @lazy
    def loss(self):
        return self.vae_loss + self.generator_loss

    @lazy
    def predictions(self):
        return self.decoder.predictions

    @get_loss_op
    def generator_loss(self):
        return (
            self.hparams.lam_c * self.attr_preserve_loss +
            self.hparams.lam_z * self.disentanglement_loss
        )

    @get_loss_op
    def attr_preserve_loss(self):
        return self.classifier_soft.loss

    @get_loss_op
    def disentanglement_loss(self):
        return tf.losses.mean_squared_error(
            labels=self.z_sample,
            predictions=self.encoder_soft.z_sample,
        )

    @property
    def eval_metrics(self):
        return self.to_metrics({
            'perplexity': self.perplexity,
            'KL': self.kl_loss,
            'recon': self.recon_loss,
            'attr_preserve': self.attr_preserve_loss,
            'disent': self.disentanglement_loss,
        })

class ClassifierModel(Model):

    @property
    def comps(self):
        return ["Classifier"]

    def init_model(self):
        self.classifier = Classifier.define( # tb Classifier
            cell=LSTMCell(self.hparams.enc_units),
            model=self,
        )

    @lazy
    def loss(self):
        return self.classifier_loss

    @lazy
    def predictions(self):
        return self.classifier.predictions

    @get_loss_op
    def classifier_loss(self):
        return self.classifier.loss

class ClassifierSynthModel(Model):

    @property
    def comps(self):
        return ["Classifier"]

    def init_model(self):
        self.generator = Decoder.define( # tb Decoder_2
            cell=LSTMCell(self.hparams.dec_units),
            model=self,
            sampling_prob=1.0,
            z_sample=self.random_z_sample(),
        )

        synth_input_embs = tf.nn.embedding_lookup(
            self.embedding_matrix,
            self.generator.predictions,
            name="synth_input_embs"
        )
        self.classifier_synth = Classifier.define( # tb Classifier_2
            cell=LSTMCell(self.hparams.enc_units),
            model=self,
            input_embs=synth_input_embs,
        )

    def random_z_sample(self):
        with tf.variable_scope("random_z"):
            batch_shape = [self.hparams.batch_size, self.hparams.z_units]
            normal = tf.distributions.Normal(
                loc=tf.zeros(batch_shape, dtype=tf.float32),
                scale=tf.ones(batch_shape, dtype=tf.float32),
                # loc=[[0.] * hparams.z_units],
                # scale=[[1.] * hparams.z_units],
                allow_nan_stats=False,
            )
            return normal.sample()

    @lazy
    def loss(self):
        return self.classify_synth_loss

    @lazy
    def predictions(self):
        return self.classifier_synth.predictions

    @get_loss_op
    def classify_synth_loss(self):
        return self.classifier_synth.loss - self.synth_class_entropy

    @get_loss_op
    def synth_class_entropy(self):
        # Empirical entropy of the classifier for synthetic data.
        # High entropy means uniformp+ class distribution; penalizing high entropy
        # encourages the model to have high confidence in predicting labels.
        log_p = self.classifier_synth.logits
        p = tf.nn.softmax(log_p)
        empirical_entropy = p * log_p
        return self.hparams.beta * tf.reduce_sum(empirical_entropy)

    @property
    def eval_metrics(self):
        return self.to_metrics({
            'synth_class_entropy': self.synth_class_entropy,
        })
