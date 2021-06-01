from distributed_ml.grad_processor.gradient_processor import GradientProcessor
from distributed_ml.grad_processor.utils import get_flattened_grads, unflatten_grads
from typing import List, Union
import torch


class TopKSparcifier(GradientProcessor):
    def __init__(self, k: Union[int, List[int]]):
        self.k = k

    @staticmethod
    def __process_flattened(grad, k):
        non_zero_idx = grad.abs().topk(k).indices
        result = torch.zeros_like(grad)
        result[non_zero_idx] = grad[non_zero_idx]
        return result

    def __call__(self, shard_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        shard_grads = [cur_layer.clone() for cur_layer in shard_grads]
        if type(self.k) is int:
            all_grads = get_flattened_grads(shard_grads)
            result = TopKSparcifier.__process_flattened(all_grads, self.k)
            assert (result != 0.).int().sum() == self.k
            unflatten_grads(shard_grads, result)
        else:
            assert type(self.k) is list
            assert len(self.k) == len(shard_grads)
            for i, (cur_k, cur_layer) in enumerate(zip(self.k, shard_grads)):
                flattened_layer = cur_layer.flatten()
                result = self.__process_flattened(flattened_layer, cur_k)
                assert (result != 0.).int().sum() == cur_k
                shard_grads[i] = result.reshape(*cur_layer.size())
        return shard_grads
