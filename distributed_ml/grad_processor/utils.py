import torch
from typing import List


def get_flattened_grads(grads: List[torch.Tensor]) -> torch.Tensor:
    """
    Babe, it's 4 pm, time for your gradient flattening
    :param grads: gradients to flatten
    :return: yes, honey
    """
    result = grads[0].flatten()
    for cur_layer in grads[1:]:
        flattened_layer = cur_layer.flatten()
        result = torch.hstack((result, flattened_layer))
    return result


def get_unflattened_grads(grads: List[torch.Tensor], all_grads: torch.Tensor) -> List[torch.Tensor]:
    start_idx = 0
    result = []
    for cur_layer in grads:
        cur_layer_size = cur_layer.numel()
        flattened_layer = all_grads[start_idx: start_idx + cur_layer_size]
        start_idx += cur_layer_size
        cur_res_layer = flattened_layer.reshape(*cur_layer.size())
        result.append(cur_res_layer)
    return result
