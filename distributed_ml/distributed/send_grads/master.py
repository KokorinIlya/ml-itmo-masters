from multiprocessing import Pipe, Process
from distributed_ml.distributed.send_grads.worker import worker
import torch
from common.evaluation import calc_accuracy
from torchvision.datasets.vision import VisionDataset
import pickle
from distributed_ml.sharding import shard_dataset, DatasetShard
import time
from typing import List, Tuple, Dict, Callable, Iterable
from multiprocessing.connection import Connection


def __check_models(base_model: torch.nn.Module, cur_model: torch.nn.Module, eps: float = 1e-4) -> bool:
    if len(list(base_model.parameters())) != len(list(cur_model.parameters())):
        return False
    for base_param, cur_param in zip(base_model.parameters(), cur_model.parameters()):
        cur_dist = (base_param - cur_param).abs().sum()
        if cur_dist > eps:
            return False
    return True


def __build_master_pipes(workers_count: int) -> List[Tuple[Connection, Connection]]:
    master_pipes = []
    for _ in range(workers_count):
        recv_chan, send_chan = Pipe(duplex=False)
        master_pipes.append((recv_chan, send_chan))
    return master_pipes


def __build_ipc_pipes(workers_count: int) -> Dict[Tuple[int, int], Tuple[Connection, Connection]]:
    result = {}
    for i in range(workers_count):
        for j in range(i + 1, workers_count):
            chan_i, chan_j = Pipe(duplex=True)
            result[(i, j)] = (chan_i, chan_j)
    return result


def __get_proc_ips_chans(ipc_pipes: Dict[Tuple[int, int], Tuple[Connection, Connection]],
                         proc_id: int, workers_count: int) -> List[Connection]:
    result = []
    for j in range(workers_count):
        if j == proc_id:
            continue
        elif j > proc_id:
            chan_i, _ = ipc_pipes[(proc_id, j)]
        else:
            assert j < proc_id
            _, chan_i = ipc_pipes[(j, proc_id)]
        result.append(chan_i)
    assert len(result) == workers_count - 1
    return result


def __build_workers(model: torch.nn.Module, workers_count: int, epochs_count: int,
                    train_batch_size: int, send_each_epoch: bool,
                    opt_getter: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
                    master_pipes: List[Tuple[Connection, Connection]],
                    ipc_pipes: Dict[Tuple[int, int], Tuple[Connection, Connection]],
                    train_shards: List[DatasetShard]) -> List[Process]:
    ps = []
    for i in range(workers_count):
        _, send_chan = master_pipes[i]
        train_shard = train_shards[i]
        p = Process(
            target=worker,
            kwargs={
                "model_bytes": pickle.dumps(model),
                "epochs_count": epochs_count,
                "train_shard": train_shard,
                "opt_getter": opt_getter,
                "train_batch_size": train_batch_size,
                "ipc_chans": __get_proc_ips_chans(ipc_pipes, proc_id=i, workers_count=workers_count),
                "master_send_chan": send_chan,
                "send_each_epoch": send_each_epoch
            }
        )
        ps.append(p)
    return ps


def master(model: torch.nn.Module, workers_count: int, epochs_count: int,
           opt_getter: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
           train_dataset: VisionDataset, train_batch_size: int,
           test_dataset: VisionDataset, test_batch_size: int = 128,
           send_each_epoch: bool = True):
    train_shards = shard_dataset(dataset=train_dataset, shards_count=workers_count, shuffle=True)
    master_pipes = __build_master_pipes(workers_count)
    ipc_pipes = __build_ipc_pipes(workers_count)
    workers = __build_workers(model=model, epochs_count=epochs_count, workers_count=workers_count,
                              opt_getter=opt_getter, send_each_epoch=send_each_epoch, train_batch_size=train_batch_size,
                              ipc_pipes=ipc_pipes, train_shards=train_shards, master_pipes=master_pipes)

    for cur_worker in workers:
        cur_worker.start()

    train_start_time = time.time()
    for epoch in range(epochs_count):
        if not send_each_epoch and epoch < epochs_count - 1:
            continue

        epoch_start_time = time.time()
        base_model = None
        for i, (recv_chan, _) in enumerate(master_pipes):
            recv_data = recv_chan.recv()
            cur_model = pickle.loads(recv_data)
            assert isinstance(cur_model, torch.nn.Module)
            if base_model is None:
                assert i == 0
                base_model = cur_model
            else:
                assert i > 0
                assert __check_models(base_model, cur_model), \
                    f"Model from worker#{i} differs from the model from worker#0"

        acc = calc_accuracy(model=model, test_dataset=test_dataset, batch_size=test_batch_size)
        cur_time = time.time()
        epoch_time_spent = int(cur_time - epoch_start_time)
        total_time_spent = int(cur_time - train_start_time)
        print(
            f"Epochs passed = {epoch + 1}, acc = {acc}, "
            f"seconds per epoch = {epoch_time_spent}, total seconds elapsed = {total_time_spent}"
        )

    for cur_worker in workers:  # TODO: close correctly (contextlib and all that stuff)
        cur_worker.join()
    for recv_chan, send_chan in master_pipes:
        recv_chan.close()
        send_chan.close()
