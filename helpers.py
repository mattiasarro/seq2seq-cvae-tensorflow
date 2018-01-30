import os
import shutil
import tensorflow as tf

def rm_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)

def lazy(fn):
    # Decorator that makes a property lazy-evaluated.
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property

def get_loss_op(fn):
    loss_name = fn.__name__

    @property
    def _scoped_loss(self):
        with tf.variable_scope(loss_name):
            loss = fn(self)
            if "losses" in self.hparams.log:
                loss = tf.Print(
                    loss,
                    [loss, self.global_step],
                    "-- (%s) %s " % (self.mode, loss_name)
                )
            return loss
    return _scoped_loss

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def pr(obj, msg=""):
    return tf.Print(obj, [obj], "-- %s --" % msg, summarize=10)

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
