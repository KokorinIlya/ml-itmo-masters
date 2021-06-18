from common.cifar import load_cifar10
from distributed_ml.sharding.model_sharding import shard_model
from common.resnet import ResNet
from distributed_ml.simulation.send_layers.train import SendLayersTrain
from distributed_ml.sharding.dataset_sharding import shard_dataset
import torch


def main():
    train_dataset = load_cifar10(is_train=True, save_path='../../../data')
    test_dataset = load_cifar10(is_train=False, save_path='../../../data')
    shards_count = 4
    shard_layers = shard_model(model=ResNet(2), shards_count=shards_count)
    train_shards = shard_dataset(dataset=train_dataset, shards_count=shards_count, mode='replicate')

    simulator = SendLayersTrain(
        epochs=20,
        train_steps=5,
        model_getter=lambda: ResNet(2),
        opt_getter=lambda params: torch.optim.SGD(params, lr=0.1, weight_decay=0.0001, momentum=0.9),
        shard_layers=shard_layers,
        train_shards=train_shards,
        test_dataset=test_dataset
    )
    simulator.train()


if __name__ == '__main__':
    main()
