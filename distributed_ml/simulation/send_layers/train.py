import time
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import DataLoader
from common.evaluation import calc_accuracy
from typing import List, Iterator, Tuple, Callable, Dict, Set
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset
from common.checks import check_models


class SendLayersTrain:
    def __init__(self,
                 epochs: int,
                 model_getter: Callable[[], torch.nn.Module],
                 opt_getter: Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer],
                 shard_layers: List[Set[str]],
                 test_dataset: VisionDataset,
                 train_shards: List[Dataset],
                 learning_batch_count: int = 5,
                 train_batch_size: int = 128,
                 test_batch_size: int = 128):
        self.models: List[torch.nn.Module] = [model_getter() for _ in range(len(shard_layers))]
        self.epochs = epochs
        self.layer_order = [name for name, _ in self.models[0].named_parameters()]
        self.opt_getter = opt_getter
        self.opts = [opt_getter(model.parameters()) for model in self.models]
        self.shard_layers = shard_layers
        self.train_shards = train_shards
        self.test_dataset = test_dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.learning_batch_count = learning_batch_count

    def __get_single_shard_weights(self, train_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]],
                                   shard_id: int) -> Tuple[Dict[str, torch.Tensor], bool]:
        cur_shard_layers = self.shard_layers[shard_id]
        model = self.models[shard_id]
        model.train()
        opt = self.opts[shard_id]
        has_modified = False

        try:
            for _ in range(self.learning_batch_count):
                X, y = next(train_iter)
                assert len(X) == len(y) and len(X) > 0

                model.zero_grad()
                for name, p in model.named_parameters():
                    p.requires_grad = name in cur_shard_layers

                y_hat = model(X)
                loss = F.cross_entropy(y_hat, y)
                loss.backward()
                opt.step()

                has_modified = True
        except StopIteration:
            pass

        cur_shard_weights = {}
        for name, weight in model.named_parameters():
            assert name not in cur_shard_weights
            if name in cur_shard_layers:
                cur_shard_weights[name] = weight

        return cur_shard_weights, has_modified

    def __collect_weights(self, train_iters: List[Iterator[Tuple[torch.Tensor, torch.Tensor]]]) -> \
            Tuple[List[Dict[str, torch.Tensor]], bool]:
        weights = []
        has_modified = False

        for shard_id, train_iter in enumerate(train_iters):
            cur_weights, cur_has_modified = self.__get_single_shard_weights(train_iter, shard_id)
            weights.append(cur_weights)
            has_modified = has_modified or cur_has_modified
        return weights, has_modified

    def __apply_weights(self, shard_weights: List[Dict[str, torch.Tensor]]) -> None:
        collected_weights = {}
        for single_shard_weights in shard_weights:
            for name, weights in single_shard_weights.items():
                assert name not in collected_weights
                collected_weights[name] = weights

        for model in self.models:
            state_dict = OrderedDict([
                (layer_name, collected_weights[layer_name].detach().clone())
                for layer_name in self.layer_order
            ])
            model.load_state_dict(state_dict, strict=False)
        self.opts = [self.opt_getter(model.parameters()) for model in self.models]

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
                weights, has_modified = self.__collect_weights(train_iters)
                if has_modified:
                    self.__apply_weights(weights)
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
