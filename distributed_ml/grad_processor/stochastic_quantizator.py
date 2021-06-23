from distributed_ml.grad_processor.gradient_processor import GradientProcessor
from distributed_ml.grad_processor.utils import get_flattened_grads, get_unflattened_grads
from typing import List
import torch


class StochasticQuantizator(GradientProcessor):
    def __init__(self, per_layer: bool = True, k: int = 4, random_bound: bool = True, seed: int = 3):
        torch.manual_seed(seed)
        self.per_layer = per_layer
        self.k = k
        self.chunk_count = 2 ** self.k
        self.random_bound = random_bound

    def __process(self, flattened_layer):
        norm = flattened_layer.square().sum().sqrt()
        one_chunk_len = 2 / self.chunk_count
        norm_layer = (flattened_layer / norm)

        if self.random_bound:
            # random (left or right) bound
            left_or_right = torch.randint(2, size=flattened_layer.size()) * one_chunk_len
        else:
            # nearest bound
            left_or_right = one_chunk_len / 2

        chunk_count_to_component = ((norm_layer + left_or_right + 1) / one_chunk_len).floor()  # from -1
        return ((chunk_count_to_component * one_chunk_len) - 1) * norm

    def __do_per_layer(self, shard_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        result = []
        for cur_layer in shard_grads:
            flattened_layer = cur_layer.flatten()
            flattened_layer = self.__process(flattened_layer)
            cur_layer_res = flattened_layer.reshape(*cur_layer.size())
            result.append(cur_layer_res)
        return result

    def __do_total(self, shard_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        all_grads = get_flattened_grads(shard_grads)
        all_grads = self.__process(all_grads)
        return get_unflattened_grads(shard_grads, all_grads)

    def __call__(self, shard_grads: List[torch.Tensor]):
        shard_grads = [cur_layer.detach().clone() for cur_layer in shard_grads]
        if self.per_layer:
            return self.__do_per_layer(shard_grads)
        else:
            return self.__do_total(shard_grads)
