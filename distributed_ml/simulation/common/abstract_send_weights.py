import time
import torch
from torch.utils.data import DataLoader
from common.evaluation import calc_accuracy
from typing import List, Iterator, Tuple, Dict, Callable
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset
from common.checks import check_models
from abc import abstractmethod
from collections import OrderedDict


class AbstractSendWeightsTrain:
    def __init__(self,
                 epochs: int,
                 models: List[torch.nn.Module],
                 opt_getter: Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer],
                 test_dataset: VisionDataset,
                 train_shards: List[Dataset],
                 train_batch_size: int = 128,
                 test_batch_size: int = 128):
        assert len(models) == len(train_shards)
        self.models = models
        self.epochs = epochs
        self.layer_order = [name for name, _ in models[0].named_parameters()]
        self.opt_getter = opt_getter
        self.opts = [self.opt_getter(model.parameters()) for model in self.models]
        self.train_shards = train_shards
        self.test_dataset = test_dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    @abstractmethod
    def _get_single_shard_weights(self, train_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]],
                                  shard_id: int) -> Tuple[Dict[str, torch.Tensor], bool]:
        pass

    def _collect_weights(self, train_iters: List[Iterator[Tuple[torch.Tensor, torch.Tensor]]]) -> \
            Tuple[List[Dict[str, torch.Tensor]], bool]:
        weights = []
        has_modified = False

        for shard_id, train_iter in enumerate(train_iters):
            cur_weights, cur_has_modified = self._get_single_shard_weights(train_iter, shard_id)
            weights.append(cur_weights)
            has_modified = has_modified or cur_has_modified
        return weights, has_modified

    @abstractmethod
    def _apply_weights(self, shard_weights: List[Dict[str, torch.Tensor]]) -> None:
        pass

    def train(self) -> List[float]:
        train_start_time = time.time()
        acc = calc_accuracy(self.models[0], self.test_dataset, batch_size=self.test_batch_size)
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

            while True:
                weights, has_modified = self._collect_weights(train_iters)
                assert len(weights) == len(self.models)
                if has_modified:
                    self._apply_weights(weights)
                else:
                    break

            base_model = self.models[0]
            for cur_model in self.models[1:]:
                assert check_models(base_model, cur_model)
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

    def _load_weights(self, collected_weights: Dict[str, torch.Tensor]):
        for model in self.models:
            state_dict = OrderedDict([
                (layer_name, collected_weights[layer_name].detach().clone())
                for layer_name in self.layer_order
            ])
            model.load_state_dict(state_dict, strict=False)
        self.opts = [self.opt_getter(model.parameters()) for model in self.models]
