import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common.evaluation import calc_accuracy
from distributed_ml.sharding import DatasetShard
from typing import List, Iterator, Tuple, Callable, Optional, Dict
from torchvision.datasets.vision import VisionDataset
from multiprocessing.connection import Connection


def send_gradients_train_proc(model: torch.nn.Module, epochs: int,
                              pipes: List[Tuple[Connection, Connection]],
                              optGetter: Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer],
                              train_shard: DatasetShard, train_batch_size: int = 128):
    pass
