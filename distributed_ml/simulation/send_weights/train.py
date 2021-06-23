import torch
import torch.nn.functional as F
from typing import List, Iterator, Tuple, Callable, Dict
from distributed_ml.simulation.common.abstract_send_weights import AbstractSendWeightsTrain
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset

from distributed_ml.simulation.common.utils import get_avg_weights


class SendWeightsTrain(AbstractSendWeightsTrain):
    def __init__(self, epochs: int, models: List[torch.nn.Module],
                 opt_getter: Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer],
                 test_dataset: VisionDataset, train_shards: List[Dataset], train_steps: int = 5,
                 train_batch_size: int = 128, test_batch_size: int = 128):
        AbstractSendWeightsTrain.__init__(self, epochs=epochs, models=models,
                                          opt_getter=opt_getter,
                                          train_shards=train_shards, test_dataset=test_dataset,
                                          train_batch_size=train_batch_size, test_batch_size=test_batch_size)
        self.train_steps = train_steps

    def _get_single_shard_weights(self, train_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]],
                                  shard_id: int) -> Tuple[Dict[str, torch.Tensor], bool]:
        model = self.models[shard_id]
        model.train()
        opt = self.opts[shard_id]
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

    @staticmethod
    def __check_weights(shard_weights: List[Dict[str, torch.Tensor]]) -> bool:
        base_weights = shard_weights[0]
        for cur_weights in shard_weights[1:]:
            if cur_weights.keys() != base_weights.keys():
                return False
            for weight_name, base_weight in base_weights.items():
                assert weight_name in cur_weights
                cur_weight = cur_weights[weight_name]
                if cur_weight.size() != base_weight.size():
                    return False
        return True

    def _apply_weights(self, shard_weights: List[Dict[str, torch.Tensor]]) -> None:
        assert SendWeightsTrain.__check_weights(shard_weights)
        collected_weights = get_avg_weights(shard_weights)

        self._load_weights(collected_weights)
