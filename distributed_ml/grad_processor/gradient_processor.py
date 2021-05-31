import torch
from typing import List
from abc import abstractmethod


class GradientProcessor:
    @abstractmethod
    def process(self, shard_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        pass
