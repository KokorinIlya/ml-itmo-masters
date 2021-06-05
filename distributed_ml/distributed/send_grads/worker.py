from multiprocessing.connection import Connection
import torch
from distributed_ml.sharding import DatasetShard
import pickle


def worker(model_bytes: bytes, epochs_count: int,
           train_shard: DatasetShard, train_batch_size: int,
           master_send_chan: Connection, send_each_epoch: bool):
    model = pickle.loads(model_bytes)
    assert isinstance(model, torch.nn.Module)
    for epoch in range(epochs_count):
        if send_each_epoch and epoch < epochs_count - 1:
            data = pickle.dumps(model)
            master_send_chan.send(data)
    data = pickle.dumps(model)
    master_send_chan.send(data)
