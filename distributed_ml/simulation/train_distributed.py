import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common.evaluation import calc_accuracy
from distributed_ml.sharding import DatasetShard
from typing import List, Iterator, Tuple, Callable
from torchvision.datasets.vision import VisionDataset


class DistributedTrain:
    def __init__(self, model: torch.nn.Module, epochs: int,
                 optGetter: Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer],
                 train_shards: List[DatasetShard], test_dataset: VisionDataset,
                 train_batch_size: int = 128, test_batch_size: int = 128):
        self.model = model
        self.epochs = epochs
        self.opt = optGetter(model.parameters())
        self.train_shards = train_shards
        self.test_dataset = test_dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    def __get_grad_single_shard(self, train_shard_iter: Iterator[Tuple[torch.Tensor,
                                                                       torch.Tensor]]) -> Tuple[List[torch.Tensor],
                                                                                                int]:
        self.model.zero_grad()
        try:
            X, y = next(train_shard_iter)
            assert len(X) == len(y) and len(X) > 0
            y_hat = self.model(X)
            loss = F.cross_entropy(y_hat, y, reduction='sum')
            loss.backward()
            return [x.grad.detach().clone() for x in self.model.parameters()], len(X)
        except StopIteration:
            return [torch.zeros_like(x) for x in self.model.parameters()], 0

    def __collect_grads(self, train_iters: List[Iterator[Tuple[torch.Tensor,
                                                               torch.Tensor]]]) -> Tuple[List[List[torch.Tensor]], int]:
        grads = []
        total_samples = 0
        for shard_iter in train_iters:
            cur_grad, samples_count = self.__get_grad_single_shard(shard_iter)
            grads.append(cur_grad)
            total_samples += samples_count
        return grads, total_samples

    def __check_sizes(self, shards_grads: List[List[torch.Tensor]], shards_count: int) -> bool:
        if len(shards_grads) != shards_count:
            return False
        params_list = list(self.model.parameters())
        for cur_shard_grads in shards_grads:
            if len(params_list) != len(cur_shard_grads):
                return False
            for cur_param_grad, cur_param in zip(cur_shard_grads, params_list):
                if cur_param_grad.size() != cur_param.size():
                    return False
        return True

    @staticmethod
    def __calc_grad(param_grads: List[torch.Tensor], param: torch.nn.Parameter,
                    shards_count: int, total_samples: int) -> torch.Tensor:
        assert len(param_grads) == shards_count
        result_grad = torch.zeros_like(param_grads[0])
        assert param.size() == result_grad.size()
        for cur_param_grad in param_grads:
            assert result_grad.size() == cur_param_grad.size()
            result_grad += cur_param_grad
        return result_grad / total_samples

    def __do_step(self, shards_grads: List[List[torch.Tensor]], shards_count: int, total_samples: int) -> None:
        assert self.__check_sizes(shards_grads, shards_count)
        for cur_param, *cur_param_grads in zip(self.model.parameters(), *shards_grads):
            result_grad = DistributedTrain.__calc_grad(cur_param_grads, cur_param, shards_count, total_samples)
            cur_param.grad = result_grad
        self.opt.step()

    def train(self) -> List[float]:
        train_start_time = time.time()
        acc = calc_accuracy(self.model, self.test_dataset, batch_size=self.test_batch_size)
        accs = [acc]
        print("Initial acc = {0}".format(acc))

        for epoch in range(self.epochs):
            self.model.train()
            epoch_start_time = time.time()

            train_iters = [
                iter(DataLoader(train_shard, batch_size=self.train_batch_size, shuffle=True))
                for train_shard in self.train_shards
            ]

            while True:
                grads, total_samples = self.__collect_grads(train_iters)
                if total_samples > 0:
                    self.__do_step(shards_grads=grads,
                                   shards_count=len(self.train_shards), total_samples=total_samples)
                else:
                    break

            acc = calc_accuracy(self.model, self.test_dataset, batch_size=self.test_batch_size)
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
