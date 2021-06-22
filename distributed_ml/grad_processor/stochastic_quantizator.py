from distributed_ml.grad_processor.gradient_processor import GradientProcessor
from distributed_ml.grad_processor.utils import get_flattened_grads, get_unflattened_grads
from typing import List
import torch
import numpy as np


class StochasticQuantizator(GradientProcessor):
    def __init__(self, per_layer: bool = True, k: int = 4, random_bound: bool = True, seed: int = 3):
        np.random.seed(seed)
        self.per_layer = per_layer
        self.k = k
        self.chunk_count = pow(2, self.k)
        self.random_bound = random_bound

    def __process(self, flattened_layer):
        norm = flattened_layer.square().sum().sqrt()
        one_chunk_len = 2 / self.chunk_count
        norm_layers = (flattened_layer / norm)

        if self.random_bound:
            # random (left or right) bound
            right = np.random.randint(2, size=flattened_layer.size()) * one_chunk_len  # left or right bounds
            return ((((norm_layers + right + 1) / one_chunk_len).floor() * one_chunk_len) - 1).float()
        else:
            # nearest bound
            return ((((norm_layers + one_chunk_len / 2 + 1) / one_chunk_len).floor() * one_chunk_len) - 1).float()

    def __call__(self, shard_grads: List[torch.Tensor]):
        shard_grads = [cur_layer.detach().clone() for cur_layer in shard_grads]

        if self.per_layer:
            for i, cur_layer in enumerate(shard_grads):
                # babe, it's 4:19 pm, time for your gradient flattening
                flattened_layer = cur_layer.flatten()  # 4:20, my darling
                flattened_layer = self.__process(flattened_layer)
                shard_grads[i] = flattened_layer.reshape(*cur_layer.size())
        else:
            all_grads = get_flattened_grads(shard_grads)
            all_grads = self.__process(all_grads)
            get_unflattened_grads(shard_grads, all_grads)

        return shard_grads
