import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common.evaluation import calc_accuracy
from distributed_ml.sharding import DatasetShard, shard_dataset
from typing import List, Iterator, Tuple, Callable, Dict
from torchvision.datasets.vision import VisionDataset
from multiprocessing.connection import Connection
from multiprocessing import Process, Pipe


def __train_epoch_proc(model: torch.nn.Module, train_loader: DataLoader,
                       opt: torch.optim.Optimizer, pipes: List[Connection]) -> None:
    for X, y in train_loader:  # TODO: send epoch end marker at all pipes after epoch is finished
        opt.zero_grad()
        y_hat = model(X)
        loss = F.cross_entropy(y_hat, y)  # TODO: sum + elements count
        loss.backward()

        shard_grads: List[torch.Tensor] = [x.grad.detach().clone() for x in model.parameters()]
        for chan in pipes:
            chan.send(shard_grads)

        other_shards_grads: List[List[torch.Tensor]] = []
        for chan in pipes:
            cur_grads = chan.recv()
            assert type(cur_grads) is list
            other_shards_grads.append(cur_grads)

        for cur_param, shard_grad, *other_grads in zip(model.parameters(), shard_grads, *other_shards_grads):
            for other_grad in other_grads:
                shard_grad += other_grad
            cur_param.grad = shard_grad
        opt.step()


def __send_gradients_train_proc(model: torch.nn.Module, epochs: int, shard_id: int,
                                pipes: List[Connection], main_pipe: Connection, send_each_epoch: bool,
                                optGetter: Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer],
                                train_shard: DatasetShard, train_batch_size: int = 128) -> None:
    opt = optGetter(model.parameters())
    train_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_start_time = time.time()

        # train_loader = DataLoader(train_shard, batch_size=train_batch_size, shuffle=True)
        # __train_epoch_proc(model, train_loader, opt, pipes)

        cur_time = time.time()
        epoch_time_spent = int(cur_time - epoch_start_time)
        total_time_spent = int(cur_time - train_start_time)
        print(
            f"Process#{shard_id}, epochs passed = {epoch + 1}, "
            f"seconds per epoch = {epoch_time_spent}, total seconds elapsed = {total_time_spent}"
        )

        if send_each_epoch and epoch < epochs - 1:
            print(f"Process#{shard_id}, sending {epoch}")
            main_pipe.send(model)

    print(f"Process#{shard_id}, last send")
    main_pipe.send(model)
    time.sleep(1)


# TODO: separate master and slave code
def __get_pipes(pipes: Dict[Tuple[int, int], Tuple[Connection, Connection]],
                shard_id: int, shards_count: int) -> List[Connection]:
    assert 0 <= shard_id < shards_count
    result = []
    for j in range(shards_count):
        if j == shard_id:
            continue
        if shard_id < j:
            i_conn, _ = pipes[(shard_id, j)]
        else:
            assert shard_id > j
            _, i_conn = pipes[(j, shard_id)]
        result.append(i_conn)
    return result


def train(model: torch.nn.Module, epochs: int, shards_count: int,
          optGetter: Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer],
          train_dataset: VisionDataset, test_dataset: VisionDataset,
          test_batch_size: int = 128, train_batch_size: int = 128) -> List[float]:
    shards = shard_dataset(dataset=train_dataset, shards_count=shards_count, shuffle=True)

    pipes = {}
    for i in range(shards_count):
        for j in range(i + 1, shards_count):
            i_conn, j_conn = Pipe(duplex=True)
            pipes[(i, j)] = (i_conn, j_conn)

    main_pipes = []
    for _ in range(shards_count):
        recv_chan, send_chan = Pipe(duplex=False)
        main_pipes.append((recv_chan, send_chan))

    processes = []
    for i in range(shards_count):
        proc_pipes = __get_pipes(pipes=pipes, shard_id=i, shards_count=shards_count)
        _, main_send_pipe = main_pipes[i]
        cur_proc = Process(
            target=__send_gradients_train_proc,
            kwargs={
                "model": model,
                "epochs": epochs,
                "shard_id": i,
                "pipes": proc_pipes,
                "main_pipe": main_send_pipe,
                "send_each_epoch": True,
                "optGetter": optGetter,
                "train_shard": shards[i],
                "train_batch_size": train_batch_size
            }
        )
        cur_proc.start()
        processes.append(cur_proc)

    # acc = calc_accuracy(model=model, test_dataset=test_dataset, batch_size=test_batch_size)
    # print(f'Initial acc = {acc}')
    accs = []  # [acc]
    train_start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        models = []
        for shard_id in range(shards_count):
            main_recv_pipe, _ = main_pipes[shard_id]
            print(f"Receiving from shard {shard_id}, epoch = {epoch}")
            cur_model = main_recv_pipe.recv()
            assert isinstance(cur_model, torch.nn.Module)
            print(f"Received from shard {shard_id}, epoch = {epoch}")
            models.append(cur_model)
        # TODO: check all models are the same
        eval_model = models[0]
        acc = calc_accuracy(model=eval_model, test_dataset=test_dataset, batch_size=test_batch_size)
        accs.append(acc)

        cur_time = time.time()
        epoch_time_spent = int(cur_time - epoch_start_time)
        total_time_spent = int(cur_time - train_start_time)
        print(
            f"Epochs passed = {epoch + 1}, acc = {acc}, "
            f"seconds per epoch = {epoch_time_spent}, total seconds elapsed = {total_time_spent}"
        )

    for cur_proc in processes:  # TODO: close correctly
        cur_proc.join()
    for i_conn, j_conn in pipes.values():
        i_conn.close()
        j_conn.close()
    for main_recv_pipe, main_send_pipe in main_pipes:
        main_recv_pipe.close()
        main_send_pipe.close()
    return accs
