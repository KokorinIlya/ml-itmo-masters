from distributed_ml.grad_processor.gradient_processor import GradientProcessor
from distributed_ml.grad_processor.utils import get_flattened_grads, unflatten_grads
from typing import List, Union
import torch


class TopKSparcifier(GradientProcessor):
    def __init__(self, k: Union[int, List[int]]):
        self.k = k

    def __call__(self, shard_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        shard_grads = [cur_layer.clone() for cur_layer in shard_grads]
        if type(self.k) is int:
            all_grads = get_flattened_grads(shard_grads)
            non_zero_idx = all_grads.abs().topk(self.k).indices
            result = torch.zeros_like(all_grads)
            result[non_zero_idx] = all_grads[non_zero_idx]
            unflatten_grads(shard_grads, result)
            return shard_grads
        else:
            assert type(self.k) is list
            assert len(self.k) == len(shard_grads)
            raise NotImplementedError()
