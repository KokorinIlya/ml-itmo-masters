import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common.evaluation import calc_accuracy
from distributed_ml.sharding import DatasetShard
from typing import List, Iterator, Tuple
from torchvision.datasets.vision import VisionDataset


def __get_grad_single_shard(model: torch.nn.Module,
                            train_shard_iter:
                            Iterator[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[List[torch.Tensor], bool]:
    model.zero_grad()
    try:
        X, y = next(train_shard_iter)
        y_hat = model(X)
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        return [x.grad.detach().clone() for x in model.parameters()], True
    except StopIteration:
        return [torch.zeros_like(x) for x in model.parameters()], False


def __collect_grads(model: torch.nn.Module,
                    train_iters: List[Iterator[Tuple[torch.Tensor,
                                                     torch.Tensor]]]) -> Tuple[List[List[torch.Tensor]], bool]:
    grads = []
    has_grads = False
    for shard_iter in train_iters:
        cur_grad, has_grad = __get_grad_single_shard(model, shard_iter)
        grads.append(cur_grad)
        has_grads = has_grads or has_grad
    return grads, has_grads


def __check_sizes(model: torch.nn.Module, shards_grads: List[List[torch.Tensor]], shards_count: int) -> bool:
    if len(shards_grads) != shards_count:
        return False
    params_list = list(model.parameters())
    for cur_shard_grads in shards_grads:
        if len(params_list) != len(cur_shard_grads):
            return False
        for cur_param_grad, cur_param in zip(cur_shard_grads, params_list):
            if cur_param_grad.size() != cur_param.size():
                return False
    return True


def __do_step(model: torch.nn.Module, opt: torch.optim.Optimizer,
              shards_grads: List[List[torch.Tensor]], shards_count: int):
    assert __check_sizes(model, shards_grads, shards_count)
    for cur_param, *cur_param_grads in zip(model.parameters(), *shards_grads):
        assert len(cur_param_grads) == shards_count
        result_grad = torch.zeros_like(cur_param_grads[0])
        for cur_param_grad in cur_param_grads:
            assert result_grad.size() == cur_param_grad.size()
            result_grad += cur_param_grad
        assert cur_param.size() == result_grad.size()
        cur_param.grad = result_grad
    opt.step()


def train_distributed(model: torch.nn.Module, epochs: int, lr: float,
                      train_shards: List[DatasetShard], test_dataset: VisionDataset,
                      train_batch_size: int = 128, test_batch_size: int = 128) -> List[float]:
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9)
    train_start_time = time.time()
    accs = []
    acc = calc_accuracy(model, test_dataset, batch_size=test_batch_size)
    print("Initial acc = {0}".format(acc))

    for epoch in range(epochs):
        model.train()
        epoch_start_time = time.time()

        train_iters = [
            iter(DataLoader(train_shard, batch_size=train_batch_size, shuffle=True))
            for train_shard in train_shards
        ]

        while True:
            grads, has_grads = __collect_grads(model, train_iters)
            if has_grads:
                __do_step(model=model, opt=opt, shards_grads=grads, shards_count=len(train_shards))
            else:
                break

        acc = calc_accuracy(model, test_dataset, batch_size=test_batch_size)
        accs.append(acc)

        cur_time = time.time()
        epoch_time_spent = int(cur_time - epoch_start_time)
        total_time_spent = int(cur_time - train_start_time)
        print(
            "Epochs passed = {0}, acc = {1}, seconds per epoch = {2}, total seconds elapsed = {3}".format(
                epoch + 1, acc, epoch_time_spent, total_time_spent
            )
        )

    return accs
