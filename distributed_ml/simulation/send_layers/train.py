import torch
import torch.nn.functional as F
from typing import List, Iterator, Tuple, Callable, Dict, Set
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset
from distributed_ml.simulation.common.abstract_send_weights import AbstractSendWeightsTrain


class SendLayersTrain(AbstractSendWeightsTrain):
    def __init__(self,
                 epochs: int,
                 models: List[torch.nn.Module],
                 opt_getter: Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer],
                 shard_layers: List[Set[str]],
                 test_dataset: VisionDataset,
                 train_shards: List[Dataset],
                 train_steps: int = 5,
                 train_batch_size: int = 128,
                 test_batch_size: int = 128):
        AbstractSendWeightsTrain.__init__(self, epochs=epochs, models=models,
                                          opt_getter=opt_getter,
                                          train_shards=train_shards, test_dataset=test_dataset,
                                          train_batch_size=train_batch_size, test_batch_size=test_batch_size)

        assert len(shard_layers) == len(train_shards)
        self.shard_layers = shard_layers
        self.train_steps = train_steps

    def _get_single_shard_weights(self, train_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]],
                                  shard_id: int) -> Tuple[Dict[str, torch.Tensor], bool]:
        cur_shard_layers = self.shard_layers[shard_id]
        model = self.models[shard_id]
        model.train()
        opt = self.opts[shard_id]
        has_modified = False

        try:
            for _ in range(self.train_steps):
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

    def _apply_weights(self, shard_weights: List[Dict[str, torch.Tensor]]) -> None:
        collected_weights = {}
        for single_shard_weights in shard_weights:
            for name, weights in single_shard_weights.items():
                assert name not in collected_weights
                collected_weights[name] = weights

        self._load_weights(collected_weights)
