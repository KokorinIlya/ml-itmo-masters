from torch.multiprocessing import Pipe, Process
from distributed_ml.distributed.send_grads.worker import worker
import torch
from common.evaluation import calc_accuracy
from torchvision.datasets.vision import VisionDataset
import pickle


def master(model: torch.nn.Module, workers_count: int, epochs_count: int,
           test_dataset: VisionDataset, test_batch_size: int = 128):
    pipes = []
    for _ in range(workers_count):
        recv_chan, send_chan = Pipe(duplex=False)
        pipes.append((recv_chan, send_chan))

    ps = []
    for _, send_chan in pipes:
        p = Process(
            target=worker,
            kwargs={
                "model": model,
                "epochs_count": epochs_count,
                "master_send_chan": send_chan
            }
        )
        p.start()
        ps.append(p)

    for epoch in range(epochs_count):
        for i, (recv_chan, _) in enumerate(pipes):
            recv_data = recv_chan.recv()
            cur_model = pickle.loads(recv_data)
            assert isinstance(cur_model, torch.nn.Module)
            cur_acc = calc_accuracy(model=model, test_dataset=test_dataset, batch_size=test_batch_size)
            print(f'{epoch + 1} epochs passed, worker#{i}, acc = {cur_acc}')
    for p in ps:
        p.join()
    for recv_chan, send_chan in pipes:
        recv_chan.close()
        send_chan.close()
