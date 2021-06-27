from common.cifar import load_cifar10
from common.resnet import ResNet
from distributed_ml.sharding.dataset_sharding import shard_dataset, ShardingMode
import torch

from distributed_ml.simulation.swarm_sgd.train import SwarmSGD


def main():
    train_dataset = load_cifar10(is_train=True, save_path='../../../data')
    test_dataset = load_cifar10(is_train=False, save_path='../../../data')
    shards_count = 4
    train_shards = shard_dataset(dataset=train_dataset, shards_count=shards_count, mode=ShardingMode.SHUFFLE_SHARD)

    simulator = SwarmSGD(
        epochs=20,
        train_steps=5,
        models=[ResNet(2) for _ in range(shards_count)],
        opt_getter=lambda params: torch.optim.SGD(params, lr=0.1, weight_decay=0.0001, momentum=0.9),
        train_shards=train_shards,
        test_dataset=test_dataset,
        group_size=2
    )
    simulator.train()


if __name__ == '__main__':
    main()
