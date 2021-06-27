from torch.multiprocessing import Process, Pipe
from common.resnet import ResNet
from common.cifar import load_cifar10
from distributed_ml.hogwild.worker import train
from common.evaluation import calc_accuracy
import torch
import time


def main():
    train_dataset = load_cifar10(is_train=True, save_path='../../data')
    test_dataset = load_cifar10(is_train=False, save_path='../../data')
    sgd_params = {"lr": 0.1, "weight_decay": 0.0001, "momentum": 0.9}
    workers_count = 2
    epochs = 10

    model = ResNet(n=2)
    acc = calc_accuracy(model, test_dataset=test_dataset, batch_size=128)
    print(f'Initial acc = {acc}')
    accs = [acc]

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == trainable_params
    print(total_params)
    model.share_memory()

    master_conns = []
    worker_conns = []
    for _ in range(workers_count):
        master_conn, worker_conn = Pipe(duplex=True)
        master_conns.append(master_conn)
        worker_conns.append(worker_conn)

    processes = [
        Process(
            target=train,
            kwargs={
                "model": model,
                "sgd_params": sgd_params,
                "epochs": epochs,
                "worker_id": worker_id,
                "train_dataset": train_dataset,
                "train_batch_size": 128,
                "master_conn": worker_conns[worker_id],
                "k": [p.numel() // 20 for p in model.parameters()],
                "use_error_correction": True
            }
        )
        for worker_id in range(workers_count)
    ]

    for p in processes:
        p.start()

    train_start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()

        for idx, conn in enumerate(master_conns):
            msg = conn.recv()
            print(f'Master has received message <<{msg}>> from worker#{idx}')

        acc = calc_accuracy(model=model, test_dataset=test_dataset, batch_size=128)
        accs.append(acc)
        cur_time = time.time()
        print(f'{epoch + 1} epochs passed, '
              f'{int(cur_time - epoch_start_time)} seconds elapsed per epoch, '
              f'{int(cur_time - train_start_time)} total seconds elapsed, '
              f'acc = {acc}')

        for conn in master_conns:
            conn.send(f'{epoch + 1} epochs passed')

    for p in processes:
        p.join()

    for cur_acc in accs:
        print(cur_acc)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    main()
