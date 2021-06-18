from multiprocessing.connection import Connection
from multiprocessing import Queue
import torch
from distributed_ml.sharding.dataset_sharding import DatasetShard
import pickle
from typing import List, Callable, Iterable, Optional
from torch.utils.data import DataLoader
import torch.nn.functional as F
from distributed_ml.distributed.send_grads.messages import Message, EndMarker, Gradient
from torchvision.datasets.vision import VisionDataset


def __check_sizes(model: torch.nn.Module, shards_grads: List[List[torch.Tensor]]) -> bool:
    params_list = list(model.parameters())
    for cur_shard_grads in shards_grads:
        if len(params_list) != len(cur_shard_grads):
            return False
        for cur_param_grad, cur_param in zip(cur_shard_grads, params_list):
            if cur_param_grad.size() != cur_param.size():
                return False
    return True


def __set_grads(model: torch.nn.Module, all_grads: List[List[torch.Tensor]], total_samples: int):
    for cur_param, *cur_param_grads in zip(model.parameters(), *all_grads):
        result_grad = torch.zeros_like(cur_param)
        for cur_param_grad in cur_param_grads:
            assert cur_param_grad.size() == cur_param.size()
            result_grad += cur_param_grad
        if total_samples > 0:
            cur_param.grad = result_grad / total_samples
        else:
            cur_param.grad.zero_()


def __update_model(model: torch.nn.Module,
                   opt: torch.optim.Optimizer,
                   recv_chans: List[Queue], is_active: List[bool],
                   own_samples: Optional[int], own_grads: Optional[List[torch.Tensor]]) -> int:
    if own_grads is not None:
        assert own_samples is not None
        all_grads = [own_grads]
        total_samples = own_samples
    else:
        assert own_samples is None
        all_grads = []
        total_samples = 0

    next_active_count = 0
    for i, cur_chan in enumerate(recv_chans):
        if is_active[i]:
            cur_msg_bytes = cur_chan.get()
            cur_msg = pickle.loads(cur_msg_bytes)
            assert isinstance(cur_msg, Message)
            if type(cur_msg) is Gradient:
                next_active_count += 1
                # noinspection PyUnresolvedReferences
                all_grads.append(cur_msg.grads)
                # noinspection PyUnresolvedReferences
                total_samples += cur_msg.samples_count
            else:
                assert type(cur_msg) is EndMarker
                is_active[i] = False
    assert __check_sizes(model, all_grads)
    __set_grads(model, all_grads, total_samples)
    opt.step()
    return next_active_count


def worker(model_bytes: bytes, epochs_count: int,
           opt_getter: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
           train_shard: DatasetShard, train_batch_size: int,
           test_dataset: VisionDataset, test_batch_size: int,
           send_chans: List[Queue], recv_chans: List[Queue],
           master_send_chan: Connection, send_each_epoch: bool):
    model = pickle.loads(model_bytes)
    assert isinstance(model, torch.nn.Module)
    opt = opt_getter(model.parameters())

    for epoch in range(epochs_count):
        assert len(send_chans) == len(recv_chans)
        is_active = [True for _ in recv_chans]
        cur_active = len(recv_chans)
        data_loader = DataLoader(dataset=train_shard, batch_size=train_batch_size, shuffle=True)

        for X, y in data_loader:
            model.train()
            assert len(X) == len(y) and len(X) > 0
            opt.zero_grad()
            y_hat = model(X)
            loss = F.cross_entropy(y_hat, y, reduction='sum')
            loss.backward()

            cur_grads = [x.grad.detach().clone() for x in model.parameters()]
            for cur_chan in send_chans:
                message = Gradient(grads=cur_grads, samples_count=len(X))
                msg_bytes = pickle.dumps(message)
                cur_chan.put(msg_bytes)
            cur_active = __update_model(model=model, opt=opt, recv_chans=recv_chans, is_active=is_active,
                                        own_grads=cur_grads, own_samples=len(X))
            # acc = calc_accuracy(model=model, test_dataset=test_dataset, batch_size=test_batch_size)
            # print(f'Worker#{os.getpid()}, acc = {acc}')

        for cur_chan in send_chans:
            message = EndMarker()
            msg_bytes = pickle.dumps(message)
            cur_chan.put(msg_bytes)

        while cur_active > 0:
            assert cur_active == sum(is_active)
            cur_active = __update_model(model=model, opt=opt, recv_chans=recv_chans, is_active=is_active,
                                        own_grads=None, own_samples=None)

        if send_each_epoch and epoch < epochs_count - 1:
            data = pickle.dumps(model)
            master_send_chan.send(data)
    data = pickle.dumps(model)
    master_send_chan.send(data)
