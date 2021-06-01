import torch
from typing import List


def get_flattened_grads(grads: List[torch.Tensor]) -> torch.Tensor:
    result = grads[0].flatten()
    for cur_layer in grads[1:]:
        flattened_layer = cur_layer.flatten()
        result = torch.hstack((result, flattened_layer))
    return result


def unflatten_grads(grads: List[torch.Tensor], all_grads: torch.Tensor) -> None:
    start_idx = 0
    for i, cur_layer in enumerate(grads):
        cur_layer_size = cur_layer.numel()
        flattened_layer = all_grads[start_idx: start_idx + cur_layer_size]
        start_idx += cur_layer_size
        grads[i] = flattened_layer.reshape(*cur_layer.size())
