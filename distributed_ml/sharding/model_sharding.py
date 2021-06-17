from typing import List, Iterator, Set
import torch
import random


def __chunk_it(seq: List[str], chunk_len: int) -> Iterator[Set[str]]:
    last = 0
    while last < len(seq):
        yield set(seq[last:last + chunk_len])
        last += chunk_len


def shard_model(model: torch.nn.Module, shards_count: int, shuffle: bool = False) -> List[Set[str]]:
    layers = [name for name, _ in model.named_parameters()]
    if shuffle:
        random.shuffle(layers)
    n = len(layers)
    layers_per_shard = n // shards_count
    if n % shards_count > 0:
        layers_per_shard += 1

    return list(__chunk_it(layers, layers_per_shard))
