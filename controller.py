import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn import learn_runner

from model import ClassifierModel, ClassifierSynthModel, VAECondModel
from hparams import hparams
from data_utils import Tokenizer, Data
import helpers

tf.logging.set_verbosity(tf.logging.DEBUG)

def get_model_class(run_config):
    if run_config.task_type in ["worker"]:
        model_name = hparams.worker_processes[run_config.task_id]
        model_class = eval(model_name)
        print("-- -- worker " + str(run_config.task_id) + " , model " + model_class.__name__)
        return model_class
    else:
        raise Exception("Didn't pick a model. Should run ps instead.")

def train(argv=None):
    run_config = tf.contrib.learn.RunConfig(
        model_dir=hparams.model_dir,
        save_checkpoints_steps=hparams.save_checkpoints_steps,
    )

    learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule="train",
        hparams=hparams
    )

# %%
def predict_str(lines):
    tokenizer = Tokenizer.get_tokenizer(hparams)
    sequences, seq_lens = tokenizer.encode_list_np(lines)
    tasks = np.array(["vae_uncond"] * sequences.shape[0])
    preds_np = predict(sequences, seq_lens, tasks)
    preds_str = tokenizer.decode(preds_np)
    return(preds_str)

def predict(sequences, seq_lens, tasks):
    run_config = tf.contrib.learn.RunConfig(
        model_dir=hparams.model_dir
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=hparams,
        config=run_config
    )

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"sequences": sequences, "seq_lens": seq_lens, "tasks": tasks},
        shuffle=False,
        batch_size=sequences.shape[0],
        num_epochs=1
    )

    preds_iterator = estimator.predict(input_fn)
    return np.stack(list(preds_iterator))

# %%
def experiment_fn(run_config, hparams):
    model_class = get_model_class(run_config)
    estimator = tf.estimator.Estimator(
        model_fn=get_model_fn(model_class),
        params=hparams,
        config=run_config
    )

    train_data = Data("train", hparams)
    eval_data = Data("eval", hparams)

    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_data.get_input_fn_for(model_class),
        eval_input_fn=eval_data.get_input_fn_for(model_class),
        train_steps=hparams.train_steps,
        eval_steps=hparams.eval_steps,
        min_eval_frequency=1, # every time checkpoint is created
        # train_monitors=[train_data.switch_dataset_hook],
        # eval_hooks=[eval_data.switch_dataset_hook],
    )

def get_model_fn(model_class):
    def model_fn(features, labels, mode, params):
        model = model_class(
            mode,
            features,
            labels,
            vocab_size=Tokenizer.get_tokenizer(hparams).vocab_size,
            hparams=params
        )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=model.predictions,
            loss=model.loss,
            train_op=model.train_op,
            eval_metric_ops=model.eval_metrics
        )
    return model_fn

mode = "train" # train | pred
if mode == "train":
    # helpers.rm_dir(hparams.model_dir)
    train_stage_names = [s for s, _ in hparams.train_stages]
    print("Train stages: " + str(train_stage_names))
    if helpers.run_from_ipython():
        try:
            tf.app.run(main=train)
        except SystemExit:
            print("-- finished --")
    elif __name__ == "__main__":
        tf.app.run(main=train)

else:
    predict_str([
        "sentence one two three Sentence",
        "sentence three sentence four",
    ])
