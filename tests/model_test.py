import tensorflow as tf
import numpy as np
import sys
import unittest

sys.path.append("..")

from hparams import hparams_unit # noqa
import helpers # noqa
from model import Model
from data_utils import Data, Tokenizer

def with_session(fn):
    def _with_session(self):
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.hook.after_create_session(sess, None)
            fn(self, sess)
    return _with_session

class ModelTest(tf.test.TestCase):

    def setUp(self):
        tf.reset_default_graph()

        self.data = Data("train", hparams_unit)
        self.hook = self.data.switch_dataset_hook
        features, labels = self.data.input_fn()
        self.model = Model(
            "train",
            features,
            labels,
            vocab_size=Tokenizer.get_tokenizer(hparams_unit).vocab_size,
            hparams=hparams_unit
        )

        # init ops
        self.model.predictions
        # self.model.loss
        self.model.train_op
        self.model.eval_metrics

        self.all_scopes = ["Encoder", "Decoder", "Classifier", "embedding"]
        self.loss_to_scope = [ # loss_name => scopes that should have a gradient
            ("loss", self.all_scopes),
            ("vae_loss", ["Encoder", "Decoder"]),
            ("generator_loss", ["Decoder"]),
            ("recon_loss", ["Encoder", "Decoder"]),
            ("kl_loss", ["Encoder"]),
            ("attr_preserve_loss", ["Decoder"]),
            ("disentanglement_loss", ["Decoder"]),
            ("classify_synth_loss", ["Classifier"])
        ]
        self.stage_changes_scopes = {
            "vae_uncond": ["Encoder", "Decoder", "embedding"],
            "classify": ["Classifier", "embedding"],
            "vae_cond_lab": ["Encoder", "Decoder", "embedding"],
            "classify_synth": ["Classifier", "embedding"],
        }
        self.fails = []

    @with_session
    def test_gradient_static(self, sess):
        for loss_name, scopes in self.loss_to_scope:
            self._assert_grad_only_in_scope(loss_name, scopes)
        self._maybe_fail()

    @with_session
    def test_gradient_runtime(self, sess):
        for task in hparams_unit.train_stage_names:
            self.hook.switch_to(task, sess)
            self._assert_correct_params_change(task, sess)
        self._maybe_fail()

    def _maybe_fail(self):
        if len(self.fails) > 0:
            msg = "\nFailed %i sub-tests:\n  " % len(self.fails)
            msg += "\n  ".join(self.fails)
            self.fail(msg)

    def _assert_grad_only_in_scope(self, loss_name, scopes):
        should_be_none = self._scopes_except(scopes)
        grads, zero_grad_vars = self._gradients(
            loss_name,
            var_list=self.model.vars_in_scopes(scopes),
        )

        def fail(msg, v):
            self.fails.append(" ".join([loss_name, msg, v.name]))

        for v in zero_grad_vars:
            scope = v.name.split("/")[0]
            if scope not in should_be_none:
                fail("SHOULD have gradient for var", v)

        for _, v in grads:
            scope = v.name.split("/")[0]
            if scope in should_be_none:
                fail("should NOT have gradient for var", v)

    def _assert_correct_params_change(self, task, sess):
        scopes = self.stage_changes_scopes[task]

        changeable = lambda: self._get_vals(scopes)
        not_changeable = lambda: self._get_vals(self._scopes_except(scopes))

        should_change = changeable()
        should_not_change = not_changeable()
        print("should_change ", len(should_change))
        print("should_not_change ", len(should_not_change))

        # print("should change")
        # for v in should_change:
        #     print(v[1])
        # 
        # print("should not change")
        # for v in should_not_change:
        #     print(v[1])
        # 
        # print("all vars")
        # for v in self._get_vars(None):
        #     print(v.name)

        self.assertEqual(
            len(should_change) + len(should_not_change),
            len(self._get_vars(None))
        )

        sess.run(self.model.train_op)
        changed = []

        for old, new in zip(should_change, changeable()):
            val_old, name = old
            val_new, _ = new
            if np.array_equal(val_old, val_new):
                self.fails.append(
                    "%s %s SHOULD change during %s" %
                    (name, str(val_old.shape), task)
                )
            else:
                changed.append(name)

        for old, new in zip(should_not_change, not_changeable()):
            val_old, name = old
            val_new, _ = new
            if not np.array_equal(val_old, val_new):
                self.fails.append(
                    "%s %s should NOT change during %s" %
                    (name, str(val_old.shape), task)
                )
                changed.append(name)
                # if task == "classify" and name == "Encoder/z/z_log_var/bias:0":
                #     print("old_val", val_old)
                #     print("new_val", val_new)

        print(task)
        print("  should change")
        for v in should_change:
            print("    " + str(v[1]))
        print("  changed")
        for v in changed:
            print("    " + v)

    def _scopes_except(self, scopes):
        return set(self.all_scopes) - set(scopes)

    def _get_vars(self, scope):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def _get_vals(self, scopes):
        return [
            (v.eval(), v.name)
            for scope in scopes
            for v in self._get_vars(scope)
        ]

    def _gradients(self, loss_name, var_list=None):
        loss = getattr(self.model, loss_name)
        grads = self.model.optimizer.compute_gradients(loss, var_list=var_list)
        ret_grads, zero_grad_vars = [], []
        for grad, var in grads:
            if grad is None:
                zero_grad_vars.append(var)
            else:
                ret_grads.append((grad, var))
        return ret_grads, zero_grad_vars

if True: # helpers.run_from_ipython():
    t = ModelTest("test_gradient_runtime")
    runner = unittest.TextTestRunner()
    runner.run(t)
else:
    tf.test.main()
