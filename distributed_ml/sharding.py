from torch.utils.data import Dataset
import random


class DatasetShard(Dataset):
    def __init__(self, dataset, indexes):
        self.__dataset = dataset
        self.__indexes = indexes

    def __getitem__(self, index):
        cur_idx = self.__indexes[index]
        return self.__dataset[cur_idx]

    def __len__(self):
        return len(self.__indexes)


def __get_shard_idx(idx, shard_id, shards_count, rows_per_shard):
    idx_begin = rows_per_shard * shard_id
    if shard_id < shards_count - 1:
        idx_end = idx_begin + rows_per_shard
    else:
        idx_end = len(idx)
    return idx[idx_begin:idx_end]


def shard_dataset(dataset, shards_count, shuffle=False):
    n = len(dataset)
    rows_per_shard = n // shards_count
    if n % shards_count > 0:
        rows_per_shard += 1
    idx = list(range(n))
    if shuffle:
        random.shuffle(idx)
    shard_idx = [__get_shard_idx(idx, i, shards_count, rows_per_shard) for i in range(shards_count)]
    return [DatasetShard(dataset=dataset, indexes=cur_idx) for cur_idx in shard_idx]
