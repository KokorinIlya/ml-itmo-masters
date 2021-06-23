from enum import Enum

from torch.utils.data import Dataset
import random
from typing import List, Generic, TypeVar
from torchvision.datasets.vision import VisionDataset

T = TypeVar('T', covariant=True)


class ShardingMode(Enum):
    SHARD = 'shard'
    SHUFFLE_SHARD = 'shuffle_shard'
    REPLICATE = 'replicate'


class DatasetShard(Dataset, Generic[T]):
    def __init__(self, dataset: Dataset, indexes: List[int]):
        self.__dataset = dataset
        self.__indexes = indexes

    def __getitem__(self, index: int) -> T:
        cur_idx = self.__indexes[index]
        return self.__dataset[cur_idx]

    def __len__(self) -> int:
        return len(self.__indexes)


def __get_shard_idx(idx: List[int], shard_id: int, shards_count: int, rows_per_shard: int) -> List[int]:
    idx_begin = rows_per_shard * shard_id
    if shard_id < shards_count - 1:
        idx_end = idx_begin + rows_per_shard
    else:
        idx_end = len(idx)
    return idx[idx_begin:idx_end]


def shard_dataset(dataset: VisionDataset, shards_count: int, mode: ShardingMode) -> List[Dataset]:
    if mode == ShardingMode.REPLICATE:
        return [dataset for _ in range(shards_count)]
    else:
        n = len(dataset)
        rows_per_shard = n // shards_count
        if n % shards_count > 0:
            rows_per_shard += 1
        idx = list(range(n))
        if mode == ShardingMode.SHUFFLE_SHARD:
            random.shuffle(idx)
        shard_idx = [__get_shard_idx(idx, i, shards_count, rows_per_shard) for i in range(shards_count)]
        return [DatasetShard(dataset=dataset, indexes=cur_idx) for cur_idx in shard_idx]
