import time
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import DataLoader
from common.evaluation import calc_accuracy
from typing import List, Iterator, Tuple, Callable, Dict
from torchvision.datasets.vision import VisionDataset


class SendWeightsTrain:
    def __init__(self,
                 epochs: int,
                 model_getter: Callable[[], torch.nn.Module],
                 opt_getter: Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer],
                 shard_layers: List[List[str]],
                 test_dataset: VisionDataset,
                 train_dataset: VisionDataset,
                 learning_batch_count: int = 5,
                 train_batch_size: int = 128,
                 test_batch_size: int = 128):
        self.models: List[torch.nn.Module] = [model_getter() for _ in range(len(shard_layers))]
        self.epochs = epochs
        self.layer_order = [name for name, _ in self.models[0].named_parameters()]
        self.opt_getter = opt_getter
        self.opts = [opt_getter(model.parameters()) for model in self.models]
        self.shard_layers = shard_layers
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.learning_batch_count = learning_batch_count

    def __get_single_shard_weights(self, train_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]],
                                   shard_id: int) -> Tuple[Dict[str, torch.Tensor], int]:
        single_shard_layers = self.shard_layers[shard_id]
        model = self.models[shard_id]
        model.train()
        opt = self.opts[shard_id]
        total_samples = 0

        try:
            for _ in range(self.learning_batch_count):
                X, y = next(train_iter)
                assert len(X) == len(y) and len(X) > 0

                model.zero_grad()
                for name, p in model.named_parameters():
                    p.requires_grad = name in single_shard_layers

                y_hat = model(X)
                loss = F.cross_entropy(y_hat, y)
                loss.backward()
                opt.step()

                total_samples += len(X)
        except StopIteration:
            pass

        single_shard_weights = {}
        for name, x in model.named_parameters():
            if name in single_shard_layers:
                single_shard_weights[name] = x

        return single_shard_weights, total_samples

    def __collect_weights(self, train_iters: List[Iterator[Tuple[torch.Tensor, torch.Tensor]]]) -> \
            Tuple[List[Dict[str, torch.Tensor]], int]:
        weights = []
        total_samples = 0

        for shard_id in range(len(train_iters)):
            train_iter = train_iters[shard_id]
            cur_weights, samples_count = self.__get_single_shard_weights(train_iter, shard_id)
            weights.append(cur_weights)
            assert total_samples == samples_count or total_samples == 0
            total_samples = samples_count
        return weights, total_samples

    def __apply_weights(self, shard_weights: List[Dict[str, torch.Tensor]]) -> None:
        collected_weights = {}
        for single_shard_weights in shard_weights:
            for name, weights in single_shard_weights.items():
                collected_weights[name] = weights

        for model in self.models:
            state_dict = OrderedDict([(layer_name, collected_weights[layer_name].detach().clone())
                                      for layer_name in self.layer_order])
            model.load_state_dict(state_dict, strict=False)
        self.opts = [self.opt_getter(model.parameters()) for model in self.models]

    def train(self) -> List[float]:
        train_start_time = time.time()
        acc = calc_accuracy(self.models[0], self.test_dataset, batch_size=self.test_batch_size)
        accs = [acc]
        print("Initial acc = {0}".format(acc))

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            for model in self.models:
                model.train()

            train_iters = [iter(DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True))
                           for _ in range(len(self.models))]

            while True:
                weights, total_samples = self.__collect_weights(train_iters)
                if total_samples > 0:
                    self.__apply_weights(weights)
                else:
                    break

            acc = calc_accuracy(self.models[0], self.test_dataset, batch_size=self.test_batch_size)
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
