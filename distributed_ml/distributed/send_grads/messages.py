import torch
from typing import List


class Message:
    pass


class EndMarker(Message):
    pass


class Gradient(Message):
    def __init__(self, grads: List[torch.Tensor], samples_count: int):
        self.grads = grads
        self.samples_count = samples_count
