from common.cifar import load_cifar10
from distributed_ml.sharding import shard_dataset
from common.resnet import ResNet
from distributed_ml.simulation.send_gradients_train import SendGradientsTrain
from distributed_ml.grad_processor.one_bit_quantizator import OneBitQuantizator
from distributed_ml.grad_processor.nop_gradient_processor import NopGradientProcessor
from distributed_ml.grad_processor.top_k_sparcifier import TopKSparcifier
import torch
import matplotlib.pyplot as plt


def main():
    train_dataset = load_cifar10(is_train=True, save_path='../../data')
    test_dataset = load_cifar10(is_train=False, save_path='../../data')
    train_shards = shard_dataset(train_dataset, shards_count=4, shuffle=True)
    model = ResNet(2)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters())
    assert total_params == trainable_params
    print(f'Params number = {total_params}')

    simulator = SendGradientsTrain(
        model=model, epochs=10,
        optGetter=lambda params: torch.optim.SGD(params, lr=0.1, weight_decay=0.0001, momentum=0.9),
        train_shards=train_shards, test_dataset=test_dataset,
        # gradient_processor=TopKSparcifier(k=[p.numel() // 5 for p in model.parameters()]),
        # gradient_processor=TopKSparcifier(k=total_params // 5),
        # gradient_processor=OneBitQuantizator(per_layer=True),
        gradient_processor=OneBitQuantizator(per_layer=False),
        train_batch_size=32,
        # save_grad_dist=True
    )
    simulator.train()
    if simulator.grad_dist is not None:
        print(simulator.grad_dist)
        plt.figure()
        plt.hist(simulator.grad_dist)
        plt.show()


if __name__ == '__main__':
    main()
