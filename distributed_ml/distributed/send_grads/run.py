from distributed_ml.distributed.send_grads.master import master
from common.cifar import load_cifar10
from common.resnet import ResNet


def main():
    test_dataset = load_cifar10(is_train=False, save_path='../../../data')
    model = ResNet(n=2)
    master(model=model, workers_count=4, epochs_count=3, test_dataset=test_dataset)


if __name__ == '__main__':
    main()
