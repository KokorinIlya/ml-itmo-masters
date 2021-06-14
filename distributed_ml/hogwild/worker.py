from typing import Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset
import time
from multiprocessing.connection import Connection


def train(model: torch.nn.Module,
          sgd_params: Dict[str, object],
          epochs: int,
          worker_id: int,
          train_dataset: VisionDataset, train_batch_size: int,
          master_conn: Connection):
    opt = torch.optim.SGD(params=model.parameters(), **sgd_params)

    for epoch in range(epochs):
        model.train()
        torch.manual_seed(time.time_ns())
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        for X, y in train_loader:
            opt.zero_grad()
            y_hat = model(X)
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            opt.step()
        master_conn.send(f'Worker#{worker_id} has finished epoch {epoch + 1}')
        msg = master_conn.recv()
        print(f'Worker#{worker_id} has received message <<{msg}>> from master')
    print(f'Worker#{worker_id} is exiting')
