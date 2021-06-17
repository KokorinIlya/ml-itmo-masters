from typing import Dict, Optional, Union, List
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset
import time
from multiprocessing.connection import Connection
from distributed_ml.grad_processor.top_k_sparcifier import TopKSparcifier


def train(model: torch.nn.Module,
          sgd_params: Dict[str, object],
          epochs: int,
          worker_id: int,
          train_dataset: VisionDataset, train_batch_size: int,
          master_conn: Connection,
          k: Optional[Union[int, List[int]]],
          use_error_correction: bool):
    if k is not None:
        grad_processor = TopKSparcifier(k)
        lr = sgd_params['lr']
    else:
        opt = torch.optim.SGD(params=model.parameters(), **sgd_params)
    err_c = [torch.zeros_like(p) for p in model.parameters()]

    for epoch in range(epochs):
        model.train()
        torch.manual_seed(time.time_ns())
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        for X, y in train_loader:
            model.zero_grad()
            y_hat = model(X)
            loss = F.cross_entropy(y_hat, y)
            loss.backward()

            if k is not None:
                grads: List[torch.Tensor] = [x.grad.detach().clone() for x in model.parameters()]

                with torch.no_grad():
                    if use_error_correction:
                        for cur_grad, cur_grad_err_c in zip(grads, err_c):
                            cur_grad += cur_grad_err_c

                    # noinspection PyUnboundLocalVariable
                    processed_grads = grad_processor(grads)

                    if use_error_correction:
                        for i, (true_grad, processed_grad) in enumerate(zip(grads, processed_grads)):
                            err_c[i] = true_grad - processed_grad

                    for param, grad in zip(model.parameters(), processed_grads):
                        # noinspection PyUnboundLocalVariable
                        new_param = param - lr * grad
                        is_zero = torch.isclose(grad, torch.zeros_like(grad))
                        param[is_zero] = param[is_zero]
                        param[~is_zero] = new_param[~is_zero]
            else:
                # noinspection PyUnboundLocalVariable
                opt.step()
        master_conn.send(f'Worker#{worker_id} has finished epoch {epoch + 1}')
        msg = master_conn.recv()
        print(f'Worker#{worker_id} has received message <<{msg}>> from master')
    print(f'Worker#{worker_id} is exiting')
