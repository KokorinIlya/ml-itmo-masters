from distributed_ml.grad_processor.gradient_processor import GradientProcessor
from distributed_ml.grad_processor.utils import get_flattened_grads, get_unflattened_grads
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

    def __do_total(self, shard_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        all_grads = get_flattened_grads(shard_grads)
        result = TopKSparcifier.__process_flattened(all_grads, self.k)
        assert (result != 0.).int().sum() == self.k
        return get_unflattened_grads(shard_grads, result)

    def __do_per_layer(self, shard_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        assert type(self.k) is list
        assert len(self.k) == len(shard_grads)
        result = []
        for cur_k, cur_layer in zip(self.k, shard_grads):
            flattened_layer = cur_layer.flatten()
            cur_layer_result = self.__process_flattened(flattened_layer, cur_k)
            assert (cur_layer_result != 0.).int().sum() == cur_k
            cur_layer_result = cur_layer_result.reshape(*cur_layer.size())
            result.append(cur_layer_result)
        return result

    def __call__(self, shard_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        shard_grads = [cur_layer.detach().clone() for cur_layer in shard_grads]
        if type(self.k) is int:
            return self.__do_total(shard_grads)
        else:
            return self.__do_per_layer(shard_grads)
