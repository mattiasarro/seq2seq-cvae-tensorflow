import tensorflow as tf

c = TF_CONFIG = {
    "task": {
        "type": "worker",
        "index": 1,
    },
    "environment": "CLOUD", # LOCAL | CLOUD
    "cluster": {
        "worker": [
            "localhost:2222",
            "localhost:2223",
            "localhost:2224"
        ],
        "ps": [
            "localhost:2220",
        ]
    }
}
cluster_spec = tf.train.ClusterSpec(c["cluster"])
ps = tf.train.Server(cluster_spec,job_name='ps')
ps.join()
