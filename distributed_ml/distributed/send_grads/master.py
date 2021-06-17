from multiprocessing import Pipe, Process, Queue
from distributed_ml.distributed.send_grads.worker import worker
import torch
from common.evaluation import calc_accuracy
from torchvision.datasets.vision import VisionDataset
import pickle
from distributed_ml.sharding.dataset_sharding import shard_dataset, DatasetShard
import time
from typing import List, Tuple, Dict, Callable, Iterable
from multiprocessing.connection import Connection
from common.checks import check_models


def __build_master_pipes(workers_count: int) -> List[Tuple[Connection, Connection]]:
    master_pipes = []
    for _ in range(workers_count):
        recv_chan, send_chan = Pipe(duplex=False)
        master_pipes.append((recv_chan, send_chan))
    return master_pipes


def __build_ipc_pipes(workers_count: int) -> Dict[Tuple[int, int], Queue]:
    result = {}
    for i in range(workers_count):
        for j in range(workers_count):
            if i == j:
                continue
            result[(i, j)] = Queue()  # i send chan, j receive chan
    return result


def __get_proc_ips_chans(ipc_pipes: Dict[Tuple[int, int], Queue],
                         proc_id: int, workers_count: int) -> Tuple[List[Queue], List[Queue]]:
    send_chans = [ipc_pipes[(proc_id, j)] for j in range(workers_count) if j != proc_id]
    assert len(send_chans) == workers_count - 1

    recv_chans = [ipc_pipes[(j, proc_id)] for j in range(workers_count) if j != proc_id]
    assert len(recv_chans) == workers_count - 1

    return send_chans, recv_chans


def __build_workers(model: torch.nn.Module, workers_count: int, epochs_count: int,
                    train_batch_size: int, send_each_epoch: bool,
                    opt_getter: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
                    master_pipes: List[Tuple[Connection, Connection]],
                    ipc_pipes: Dict[Tuple[int, int], Queue],
                    train_shards: List[DatasetShard],
                    test_dataset: VisionDataset, test_batch_size: int) -> List[Process]:
    ps = []
    for i in range(workers_count):
        _, master_send_chan = master_pipes[i]
        train_shard = train_shards[i]
        send_chans, recv_chans = __get_proc_ips_chans(ipc_pipes, proc_id=i, workers_count=workers_count)
        p = Process(
            target=worker,
            kwargs={
                "model_bytes": pickle.dumps(model),
                "epochs_count": epochs_count,
                "train_shard": train_shard,
                "opt_getter": opt_getter,
                "train_batch_size": train_batch_size,
                "test_dataset": test_dataset,
                "test_batch_size": test_batch_size,
                "send_chans": send_chans,
                "recv_chans": recv_chans,
                "master_send_chan": master_send_chan,
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
                              opt_getter=opt_getter, send_each_epoch=send_each_epoch,
                              train_batch_size=train_batch_size,
                              ipc_pipes=ipc_pipes, train_shards=train_shards, master_pipes=master_pipes,
                              test_dataset=test_dataset, test_batch_size=test_batch_size)

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
                assert check_models(base_model, cur_model), \
                    f"Model from worker#{i} differs from the model from worker#0"

        acc = calc_accuracy(model=base_model, test_dataset=test_dataset, batch_size=test_batch_size)
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
