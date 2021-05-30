from common.cifar import load_cifar10
from distributed_ml.sharding import shard_dataset
from common.resnet import ResNet
from distributed_ml.simulation.train_distributed import train_distributed


def main():
    train_dataset = load_cifar10(is_train=True, save_path='../data')
    test_dataset = load_cifar10(is_train=False, save_path='../data')
    train_shards = shard_dataset(train_dataset, shards_count=1, shuffle=False)
    model = ResNet(2)
    train_distributed(model=model, epochs=10, train_shards=train_shards, test_dataset=test_dataset)


if __name__ == '__main__':
    main()
