import tensorflow as tf
from copy import deepcopy

pre = "./"
save_checkpoints_steps = 3
train_stages = [ # name, num_steps (multiple of save_checkpoints_steps so we always evaluate after processing one stage)
    ["vae_uncond", 1 * save_checkpoints_steps],
    ["classify", 1 * save_checkpoints_steps],
    ["vae_cond_lab", 1 * save_checkpoints_steps],
    ["classify_synth", 1 * save_checkpoints_steps],
]
train_steps = sum([steps for n, steps in train_stages])
train_steps += len(train_stages) - 1 # because we increment global_step before calculating loss

train_steps = 1000

def smallest_match(match_num, input):
    sorted_keys = sorted(input.keys())
    for num in sorted_keys:
        if match_num <= num:
            return input[num]
    return input[sorted_keys[-1]]

schedyled_sampling_k = smallest_match(train_steps, {
    10000: 1000,
    5000: 500,
    100: 10,
    20: 2,
    3: 1,
})
kl_weight = smallest_match(train_steps, {
    20: {"multiplier": 2, "offset": 10},
    10: {"multiplier": 10, "offset": -10}, # KL loss is nearly 1 immediately
})

hparams = tf.contrib.training.HParams(
    dataset="imdb", # imdb | unit_test
    model_dir=pre + 'models/imdb',
    data_dir=pre + 'data/processed',
    train_stages=train_stages,
    train_stage_names=[s for s, _ in train_stages],
    save_checkpoints_steps=save_checkpoints_steps,
    log=["losses", "tasks"], # "losses", "tasks"
    worker_processes=[
        "VAECondModel", # /job:worker/task:0
        "ClassifierModel", # /job:worker/task:1
        "ClassifierSynthModel", # /job:worker/task:2
    ],

    variational=True,
    learning_rate=0.002,
    num_classes=2,

    train_steps=train_steps,
    eval_steps=2,
    scheduled_sampling_k=schedyled_sampling_k,
    kl_weight_multiplier=kl_weight["multiplier"],
    kl_weight_offset=kl_weight["offset"],

    buffer_size=10000,
    batch_size=10,
    max_seq_len=31, # when decoding; +1 to allow for EOS token

    embedding_size=10,
    enc_units=10,
    dec_units=20,
    z_units=21,

    lam_c=1, # coefficient for attribute preserve loss
    lam_z=1, # coefficient for disentaglement loss
    beta=1.0, # beta in formula 10: weight on the loss of synthetic data classification entropy
)

hparams_unit = deepcopy(hparams)
hparams_unit.data_dir = pre + 'data/unit_test'
hparams_unit.dataset = 'unit_test'
hparams_unit.batch_size = 2
