import json

TF_CONFIG = {
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

json.dumps(TF_CONFIG)
