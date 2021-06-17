from common.cifar import load_cifar10
from distributed_ml.sharding.model_sharding import shard_model
from common.resnet import ResNet
from distributed_ml.simulation.layers.train import SendWeightsTrain
import torch


def main():
    train_dataset = load_cifar10(is_train=True, save_path='../../../data')
    test_dataset = load_cifar10(is_train=False, save_path='../../../data')
    model = ResNet(2)
    shard_layers = shard_model(model, shards_count=4)

    simulator = SendWeightsTrain(
        epochs=20,
        learning_batch_count=5,
        model_getter=lambda: ResNet(2),
        opt_getter=lambda params: torch.optim.SGD(params, lr=0.1, weight_decay=0.0001, momentum=0.9),
        shard_layers=shard_layers,
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )
    simulator.train()


if __name__ == '__main__':
    main()
