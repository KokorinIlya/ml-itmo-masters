from distributed_ml.grad_processor.gradient_processor import GradientProcessor
from typing import List
import torch


class NopGradientProcessor(GradientProcessor):
    def __call__(self, shard_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        return shard_grads
