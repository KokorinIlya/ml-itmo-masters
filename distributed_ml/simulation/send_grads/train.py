import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common.evaluation import calc_accuracy
from torch.utils.data import Dataset
from typing import List, Iterator, Tuple, Callable, Optional
from distributed_ml.grad_processor.gradient_processor import GradientProcessor
from distributed_ml.grad_processor.nop_gradient_processor import NopGradientProcessor
from torchvision.datasets.vision import VisionDataset


class SendGradientsTrain:
    def __init__(self, model: torch.nn.Module, epochs: int,
                 opt_getter: Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer],
                 train_shards: List[Dataset], test_dataset: VisionDataset,
                 gradient_processor: GradientProcessor = NopGradientProcessor(),
                 use_error_correction: bool = False,
                 train_batch_size: int = 128, test_batch_size: int = 128,
                 save_grad_dist: bool = False):
        self.model = model
        self.epochs = epochs
        self.opt = opt_getter(model.parameters())
        self.train_shards = train_shards
        self.test_dataset = test_dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.gradient_processor = gradient_processor

        if use_error_correction:
            self.error_correction: Optional[List[List[torch.Tensor]]] = [
                [torch.zeros_like(x) for x in model.parameters()]
                for _ in range(len(train_shards))
            ]
        else:
            self.error_correction: Optional[List[List[torch.Tensor]]] = None

        self.grad_dist: Optional[List[float]] = [] if save_grad_dist else None
        self.X: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None

    def __add_X_y(self, X, y):
        if self.grad_dist is not None:
            if self.X is None:
                assert self.y is None
                self.X = X
                self.y = y
            else:
                assert self.y is not None
                assert len(self.y) == len(self.X)
                assert self.X.size()[1:] == X.size()[1:]
                self.X = torch.vstack((self.X, X))
                self.y = torch.hstack((self.y, y))
        else:
            assert self.X is None and self.y is None

    def __get_grad_single_shard(self, shard_id: int,
                                train_shard_iter: Iterator[Tuple[torch.Tensor,
                                                                 torch.Tensor]]) -> Tuple[List[torch.Tensor], int]:
        self.model.zero_grad()
        try:
            X, y = next(train_shard_iter)
            assert len(X) == len(y) and len(X) > 0
            y_hat = self.model(X)
            loss = F.cross_entropy(y_hat, y, reduction='sum')
            self.__add_X_y(X, y)
            loss.backward()

            shard_grads: List[torch.Tensor] = [x.grad.detach().clone() for x in self.model.parameters()]

            if self.error_correction is not None:
                cur_shard_err_c = self.error_correction[shard_id]
                assert len(cur_shard_err_c) == len(shard_grads)
                for cur_param_grad, cur_param_err_c in zip(shard_grads, cur_shard_err_c):
                    cur_param_grad += cur_param_err_c

                processed_grads = self.gradient_processor(shard_grads)
                assert len(processed_grads) == len(shard_grads)

                for i, (cur_param_init_grad, cur_param_res_grad) in enumerate(zip(shard_grads, processed_grads)):
                    cur_shard_err_c[i] = cur_param_init_grad - cur_param_res_grad
                return processed_grads, len(X)
            else:
                return self.gradient_processor(shard_grads), len(X)
        except StopIteration:
            return [torch.zeros_like(x) for x in self.model.parameters()], 0

    def __collect_grads(self,
                        train_iters: List[Iterator[Tuple[torch.Tensor,
                                                         torch.Tensor]]]) -> Tuple[List[List[torch.Tensor]], int]:
        grads = []
        total_samples = 0
        for shard_id, shard_iter in enumerate(train_iters):
            cur_grad, samples_count = self.__get_grad_single_shard(shard_id, shard_iter)
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

    def __get_true_grad(self) -> List[torch.Tensor]:
        assert len(self.X) == len(self.y)
        self.model.zero_grad()
        y_hat = self.model(self.X)
        loss = F.cross_entropy(y_hat, self.y)
        loss.backward()
        return [x.grad.detach().clone() for x in self.model.parameters()]

    def __do_step(self, shards_grads: List[List[torch.Tensor]], shards_count: int, total_samples: int) -> None:
        assert self.__check_sizes(shards_grads, shards_count)

        if self.grad_dist is None:
            assert self.X is None and self.y is None
            for cur_param, *cur_param_grads in zip(self.model.parameters(), *shards_grads):
                result_grad = SendGradientsTrain.__calc_grad(cur_param_grads, cur_param, shards_count, total_samples)
                cur_param.grad = result_grad
        else:
            assert self.X is not None and self.y is not None
            true_grad = self.__get_true_grad()
            total_dist = 0.
            total_count = 0
            for cur_param, cur_param_true_grad, *cur_param_grads in zip(self.model.parameters(),
                                                                        true_grad,
                                                                        *shards_grads):
                result_grad = SendGradientsTrain.__calc_grad(cur_param_grads, cur_param, shards_count, total_samples)

                assert cur_param_true_grad.size() == result_grad.size()
                dist_m = (result_grad - cur_param_true_grad)  # / (cur_param_true_grad + 1e-3)
                total_dist += dist_m.abs().sum().item()
                total_count += result_grad.numel()

                cur_param.grad = result_grad
            assert total_count > 0
            self.grad_dist.append(total_dist / total_count)
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
                self.X = None
                self.y = None
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
