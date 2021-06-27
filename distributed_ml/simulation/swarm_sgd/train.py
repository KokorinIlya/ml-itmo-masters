import random
import time
from typing import List, Callable, Iterator, Tuple, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VisionDataset

from common.checks import check_models
from common.evaluation import calc_accuracy
from distributed_ml.simulation.common.utils import get_avg_weights


class SwarmSGD:
    def __init__(self, epochs: int, models: List[torch.nn.Module],
                 opt_getter: Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer],
                 test_dataset: VisionDataset, train_shards: List[Dataset], train_steps: int = 5,
                 train_batch_size: int = 128, test_batch_size: int = 128, group_size: int = 3):
        self.models = models
        self.epochs = epochs
        self.opt_getter = opt_getter
        self.train_shards = train_shards
        self.test_dataset = test_dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_steps = train_steps
        self.group_size = group_size
        self.shards_ids = list(range(len(models)))
        assert len(models) == len(train_shards)
        assert len(train_shards) % group_size == 0

    def __train_single_shard(self, train_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]],
                             shard_id: int) -> Tuple[Dict[str, torch.Tensor], bool]:
        model = self.models[shard_id]
        opt = self.opts[shard_id]
        model.train()
        has_modified = False
        try:
            for _ in range(self.train_steps):
                X, y = next(train_iter)
                assert len(X) == len(y) and len(X) > 0

                model.zero_grad()
                y_hat = model(X)
                loss = F.cross_entropy(y_hat, y)
                loss.backward()
                opt.step()

                has_modified = True
        except StopIteration:
            pass

        return dict(model.named_parameters()), has_modified

    def __apply_weights(self, start_shard: int, end_shard: int, avg_weights: Dict[str, torch.Tensor]):
        for shard in range(start_shard, end_shard):
            model = self.models[self.shards_ids[shard]]
            state_dict = {name: weight.detach().clone() for name, weight in avg_weights.items()}
            model.load_state_dict(state_dict, strict=False)

    def __process_group(self, start_shard: int, end_shard: int,
                        train_iters: List[Iterator[Tuple[torch.Tensor, torch.Tensor]]]) -> bool:
        train_results = [self.__train_single_shard(train_iters[self.shards_ids[shard]], self.shards_ids[shard]) for
                         shard in range(start_shard, end_shard)]
        train_weights = [shard_weights for shard_weights, _ in train_results]
        has_modified = any([shard_changed for _, shard_changed in train_results])
        avg_weights = get_avg_weights(train_weights)
        self.__apply_weights(start_shard, end_shard, avg_weights)
        return has_modified

    def __do_step(self, train_iters: List[Iterator[Tuple[torch.Tensor, torch.Tensor]]]) -> bool:
        has_modified = False
        random.shuffle(self.shards_ids)
        self.opts = [self.opt_getter(model.parameters()) for model in self.models]
        for start_shard in range(0, len(self.shards_ids), self.group_size):
            has_modified = has_modified or self.__process_group(start_shard, start_shard + self.group_size, train_iters)
        return has_modified

    def train(self) -> List[float]:
        train_start_time = time.time()
        base_model = self.models[0]
        acc = calc_accuracy(base_model, self.test_dataset, batch_size=self.test_batch_size)
        accs = [acc]
        print(f'Initial acc = {acc}')

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            for model in self.models:
                model.train()

            train_iters = [
                iter(DataLoader(cur_train_shard, batch_size=self.train_batch_size, shuffle=True))
                for cur_train_shard in self.train_shards
            ]

            has_changes = True
            while has_changes:
                has_changes = self.__do_step(train_iters)

            all_weights = [dict(model.named_parameters()) for model in self.models]
            avg_weighs = get_avg_weights(all_weights)
            self.__apply_weights(0, len(self.shards_ids), avg_weighs)  # Global exchange
            for i in range(1, len(self.models)):
                assert check_models(base_model, self.models[i]), f'{i}-th model differs from base (0-th)'
            acc = calc_accuracy(base_model, self.test_dataset, batch_size=self.test_batch_size)
            accs.append(acc)

            cur_time = time.time()
            epoch_time_spent = int(cur_time - epoch_start_time)
            total_time_spent = int(cur_time - train_start_time)
            print(
                f'Epochs passed = {epoch + 1}, acc = {acc}, '
                f'seconds per epoch = {epoch_time_spent}, total seconds elapsed = {total_time_spent}'
            )

        return accs
