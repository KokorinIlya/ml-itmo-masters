import copy
import time
from typing import Callable, Iterator, List, Tuple, Dict

import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from common.evaluation import calc_accuracy
from distributed_ml.sharding import DatasetShard
from distributed_ml.utils import check_models


class SendParametersTrain:
    def __init__(self, model_getter: Callable[[], torch.nn.Module], epochs: int,
                 opt_getter: Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer],
                 train_shards: List[DatasetShard], test_dataset: VisionDataset,
                 train_batch_size: int = 128, test_batch_size: int = 128,
                 batches_pers_step: int = 5):
        self.epochs = epochs
        self.models: List[torch.nn.Module] = [model_getter() for _ in range(len(train_shards))]
        self.opts: List[torch.optim.Optimizer] = [opt_getter(cur_model.parameters()) for cur_model in self.models]
        self.train_shards = train_shards
        self.test_dataset = test_dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.batches_pers_step = batches_pers_step * 2
        self.steps_per_epoch = max(
            list(map(lambda shard: shard.batch_steps_cnt(train_batch_size, batches_pers_step), train_shards))
        )

    def __get_params_single_shard(self, shard_id: int, train_shard_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]]) \
            -> Dict[str, torch.Tensor]:
        model = self.models[shard_id]
        opt = self.opts[shard_id]
        model.train()
        try:
            for _ in range(self.batches_pers_step):
                opt.zero_grad()
                X, y = next(train_shard_iter)
                assert len(X) == len(y) and len(X) > 0
                y_hat = model(X)
                loss = F.cross_entropy(y_hat, y)
                loss.backward()
                opt.step()
        except StopIteration:
            pass
        return dict(model.named_parameters())

    @staticmethod
    def __get_avg_param(name: str, shards_params: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        needed_params = [shard_params[name] for shard_params in shards_params]
        with torch.no_grad():
            return torch.mean(input=torch.stack(needed_params), dim=0)

    def __do_step(self, train_iters: List[Iterator[Tuple[torch.Tensor, torch.Tensor]]]):
        shards_params = [self.__get_params_single_shard(shard_id, train_iter) for shard_id, train_iter in
                         enumerate(train_iters)]

        avg_params = {name: SendParametersTrain.__get_avg_param(name, shards_params) for name in
                      shards_params[0].keys()}
        for model in self.models:
            copy_avg = {name: param.detach().clone() for name, param in avg_params.items()}
            model.load_state_dict(copy_avg, strict=False)
        for i in range(1, len(self.models)):
            assert check_models(self.models[0], self.models[i]), f'models 0 and {i} differ'

    def train(self) -> List[float]:
        train_start_time = time.time()
        self.models[0].eval()
        accs = [calc_accuracy(self.models[0], self.test_dataset, batch_size=self.test_batch_size)]
        print(f"Initial acc = {accs[0]}")
        for epoch in range(1, self.epochs + 1):  # to get 1 based indexing
            epoch_start_time = time.time()
            train_iters = [iter(DataLoader(train_shard, batch_size=self.train_batch_size, shuffle=True)) for train_shard
                           in self.train_shards]
            for i in range(self.steps_per_epoch):
                self.__do_step(train_iters)
            cur_accs = [calc_accuracy(model.eval(), self.test_dataset, batch_size=self.test_batch_size) for model in
                        self.models]
            accs.append(cur_accs[0])
            epoch_end_time = time.time()
            epoch_time_spent = int(epoch_end_time - epoch_start_time)
            total_time_spent = int(epoch_end_time - train_start_time)
            print(f"Epochs passed = {epoch}, accs = {cur_accs}, ", end='')
            print(f"seconds per epoch = {epoch_time_spent}, total seconds elapsed = {total_time_spent}")
        return accs
