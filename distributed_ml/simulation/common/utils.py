from typing import List, Dict

import torch


def get_avg_weights(shards_weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    avg_weights = {}
    with torch.no_grad():
        base_weights = shards_weights[0]
        for cur_weight_name in base_weights.keys():
            result_weight = torch.zeros_like(base_weights[cur_weight_name])
            for cur_shard_weights in shards_weights:
                result_weight += cur_shard_weights[cur_weight_name]
            avg_weights[cur_weight_name] = result_weight / len(shards_weights)
    return avg_weights
