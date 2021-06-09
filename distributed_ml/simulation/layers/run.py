from common.cifar import load_cifar10
from distributed_ml.sharding_model import shard_model
from common.resnet import ResNet
from distributed_ml.simulation.layers.train import SendGradientsTrain
import torch
import matplotlib.pyplot as plt


def main():
    train_dataset = load_cifar10(is_train=True, save_path='../../../data')
    test_dataset = load_cifar10(is_train=False, save_path='../../../data')
    model = ResNet(2)
    if torch.cuda.is_available():
        model.cuda()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters())
    assert total_params == trainable_params

    shard_layers = shard_model(model, shards_count=4)

    simulator = SendGradientsTrain(
        model=model, epochs=5,
        optGetter=lambda params: torch.optim.SGD(params, lr=0.1, weight_decay=0.0001, momentum=0.9),
        shard_layers=shard_layers, train_dataset=train_dataset, test_dataset=test_dataset,
        train_batch_size=32,
        save_grad_dist=False
    )
    simulator.train()
    # if simulator.grad_dist is not None:
    #     print(simulator.grad_dist)
    #     plt.figure()
    #     plt.hist(simulator.grad_dist)
    #     plt.savefig('../../../plots/grad_loss_none.png')


if __name__ == '__main__':
    main()
