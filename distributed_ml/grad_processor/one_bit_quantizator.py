from distributed_ml.grad_processor.gradient_processor import GradientProcessor
from distributed_ml.grad_processor.utils import get_flattened_grads, unflatten_grads
from typing import List
import torch


class OneBitQuantizator(GradientProcessor):
    def __init__(self, per_layer: bool):
        self.per_layer = per_layer

    @staticmethod
    def __process_flattened(flattened_layer):
        mean_positive = flattened_layer[flattened_layer >= 0].mean()
        mean_negative = flattened_layer[flattened_layer < 0].mean()
        flattened_layer[flattened_layer >= 0] = mean_positive
        flattened_layer[flattened_layer < 0] = mean_negative

    def __call__(self, shard_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        shard_grads = [cur_layer.clone() for cur_layer in shard_grads]
        if self.per_layer:
            for i, cur_layer in enumerate(shard_grads):
                # babe, it's 4 pm, time for your gradient flattening
                flattened_layer = cur_layer.flatten()  # yes, honey
                OneBitQuantizator.__process_flattened(flattened_layer)
                assert flattened_layer.unique().size() == 2
                shard_grads[i] = flattened_layer.reshape(*cur_layer.size())
        else:
            all_grads = get_flattened_grads(shard_grads)
            OneBitQuantizator.__process_flattened(all_grads)
            assert all_grads.unique().size() == 2
            unflatten_grads(shard_grads, all_grads)

        return shard_grads
