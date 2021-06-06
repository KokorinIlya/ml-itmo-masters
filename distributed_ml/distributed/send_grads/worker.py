from multiprocessing.connection import Connection
import torch
from distributed_ml.sharding import DatasetShard
import pickle
from typing import List, Callable, Iterable
from torch.utils.data import DataLoader


def worker(model_bytes: bytes, epochs_count: int,
           opt_getter: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
           train_shard: DatasetShard, train_batch_size: int,
           ipc_chans: List[Connection],
           master_send_chan: Connection, send_each_epoch: bool):
    model = pickle.loads(model_bytes)
    assert isinstance(model, torch.nn.Module)
    opt = opt_getter(model.parameters())

    for epoch in range(epochs_count):
        is_active = [True for _ in ipc_chans]
        active_count = len(ipc_chans)
        data_loader = DataLoader(dataset=train_shard, batch_size=train_batch_size, shuffle=True)

        if send_each_epoch and epoch < epochs_count - 1:
            data = pickle.dumps(model)
            master_send_chan.send(data)
    data = pickle.dumps(model)
    master_send_chan.send(data)
