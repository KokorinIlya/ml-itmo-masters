from typing import List
import torch


def get_layer(model: torch.nn.Module, layer_name: str) -> torch.Tensor:
    names = layer_name.split('.')
    res = model
    for name in names:
        res = getattr(res, name)
    return res


def chunk_it(seq: List, chunk_len: int) -> List[List]:
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + chunk_len)])
        last += chunk_len

    return out


def shard_model(model: torch.nn.Module, shards_count: int, shuffle: bool = False):
    # layers = [get_layer(model, name) for name, _ in model.named_parameters()]
    layers = [name for name, _ in model.named_parameters()]
    n = len(layers)
    layers_per_shard = n // shards_count
    if n % shards_count > 0:
        layers_per_shard += 1

    shard_layers = chunk_it(layers, layers_per_shard)
    return shard_layers
