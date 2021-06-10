import torch
from matplotlib import pyplot as plt

from common.cifar import load_cifar10
from common.resnet import ResNet
from distributed_ml.sharding import shard_dataset
from distributed_ml.simulation.send_parameters.train import SendParametersTrain


def main():
    train_dataset = load_cifar10(is_train=True, save_path='../../../data')
    test_dataset = load_cifar10(is_train=False, save_path='../../../data')
    train_shards = shard_dataset(train_dataset, shards_count=4, shuffle=True)
    model = ResNet(2)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters())
    assert total_params == trainable_params
    print(f'Params number = {total_params}')

    simulator = SendParametersTrain(
        model_getter=lambda: ResNet(2), epochs=10,
        opt_getter=lambda params: torch.optim.SGD(params, lr=0.1, weight_decay=0.0001, momentum=0.9),
        train_shards=train_shards, test_dataset=test_dataset,
        train_batch_size=32,
    )
    res = simulator.train()
    print(res)


if __name__ == '__main__':
    main()
