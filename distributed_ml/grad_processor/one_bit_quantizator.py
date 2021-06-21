from distributed_ml.grad_processor.gradient_processor import GradientProcessor
from distributed_ml.grad_processor.utils import get_flattened_grads, get_unflattened_grads
from typing import List
import torch


class OneBitQuantizator(GradientProcessor):
    def __init__(self, per_layer: bool):
        self.per_layer = per_layer

    @staticmethod
    def __process(grad: torch.Tensor):
        mean_positive = grad[grad >= 0].mean()
        mean_negative = grad[grad < 0].mean()
        grad[grad >= 0] = mean_positive
        grad[grad < 0] = mean_negative

    @staticmethod
    def __do_per_layer(shard_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        result = []
        for cur_layer in shard_grads:
            OneBitQuantizator.__process(cur_layer)
            assert cur_layer.unique().numel() <= 2
            cur_layer_res = cur_layer.reshape(*cur_layer.size())
            result.append(cur_layer_res)
        return result

    @staticmethod
    def __do_total(shard_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        all_grads = get_flattened_grads(shard_grads)
        OneBitQuantizator.__process(all_grads)
        assert all_grads.unique().numel() <= 2
        return get_unflattened_grads(shard_grads, all_grads)

    def __call__(self, shard_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        shard_grads = [cur_layer.detach().clone() for cur_layer in shard_grads]
        if self.per_layer:
            return OneBitQuantizator.__do_per_layer(shard_grads)
        else:
            return OneBitQuantizator.__do_total(shard_grads)
