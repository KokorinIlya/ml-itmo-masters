from distributed_ml.distributed.send_grads.master import master
from common.cifar import load_cifar10
from common.resnet import ResNet
import torch


def main():
    test_dataset = load_cifar10(is_train=False, save_path='../../../data')
    train_dataset = load_cifar10(is_train=True, save_path='../../../data')
    model = ResNet(n=2)
    master(model=model, workers_count=2, epochs_count=10,
           opt_getter=lambda params: torch.optim.SGD(params, lr=0.1, weight_decay=0.0001, momentum=0.9),
           train_dataset=train_dataset, train_batch_size=128,
           test_dataset=test_dataset,
           send_each_epoch=True)


if __name__ == '__main__':
    main()
