from common.cifar import load_cifar10
from common.resnet import ResNet
from distributed_ml.distributed.send_gradients_train import train
import torch


def main():
    train_dataset = load_cifar10(is_train=True, save_path='../../data')
    test_dataset = load_cifar10(is_train=False, save_path='../../data')
    model = ResNet(2)
    train(
        model=model, epochs=2, shards_count=2,
        optGetter=lambda params: torch.optim.SGD(params, lr=0.1, weight_decay=0.0001, momentum=0.9),
        train_dataset=train_dataset, test_dataset=test_dataset,
        train_batch_size=64
    )


if __name__ == '__main__':
    main()
