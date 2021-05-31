from distributed_ml.grad_processor.gradient_processor import GradientProcessor
from typing import List
import torch


class OneBitQuantizator(GradientProcessor):
    def __init__(self, per_layer: bool):
        self.per_layer = per_layer

    def process(self, shard_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        raise NotImplementedError()
