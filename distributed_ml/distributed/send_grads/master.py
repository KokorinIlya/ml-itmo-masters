from torch.multiprocessing import Pipe, Process
from distributed_ml.distributed.send_grads.worker import worker
import torch
from common.evaluation import calc_accuracy
from torchvision.datasets.vision import VisionDataset
import pickle
from distributed_ml.sharding import shard_dataset


def __check_models(base_model: torch.nn.Module, cur_model: torch.nn.Module, eps: float = 1e-4) -> bool:
    if len(list(base_model.parameters())) != len(list(cur_model.parameters())):
        return False
    for base_param, cur_param in zip(base_model.parameters(), cur_model.parameters()):
        cur_dist = (base_param - cur_param).abs().sum()
        if cur_dist > eps:
            return False
    return True


def master(model: torch.nn.Module, workers_count: int, epochs_count: int,
           train_dataset: VisionDataset, train_batch_size: int,
           test_dataset: VisionDataset, test_batch_size: int = 128):
    train_shards = shard_dataset(dataset=train_dataset, shards_count=workers_count, shuffle=True)

    pipes = []
    for _ in range(workers_count):
        recv_chan, send_chan = Pipe(duplex=False)
        pipes.append((recv_chan, send_chan))

    ps = []
    for i in range(workers_count):
        _, send_chan = pipes[i]
        train_shard = train_shards[i]
        p = Process(
            target=worker,
            kwargs={
                "model_bytes": pickle.dumps(model),
                "epochs_count": epochs_count,
                "train_shard": train_shard,
                "train_batch_size": train_batch_size,
                "master_send_chan": send_chan,
                "send_each_epoch": True
            }
        )
        p.start()
        ps.append(p)

    for epoch in range(epochs_count):
        base_model = None
        for i, (recv_chan, _) in enumerate(pipes):
            recv_data = recv_chan.recv()
            cur_model = pickle.loads(recv_data)
            assert isinstance(cur_model, torch.nn.Module)
            if base_model is None:
                assert i == 0
                base_model = cur_model
            else:
                assert i > 0
                assert __check_models(base_model, cur_model), \
                    f"Model from worker#{i} differs from the model from worker#0"

        acc = calc_accuracy(model=model, test_dataset=test_dataset, batch_size=test_batch_size)
        print(f'{epoch + 1} epochs passed, acc = {acc}')
    for p in ps:  # TODO: close correctly
        p.join()
    for recv_chan, send_chan in pipes:
        recv_chan.close()
        send_chan.close()
