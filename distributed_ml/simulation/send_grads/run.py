from common.cifar import load_cifar10
from distributed_ml.sharding.dataset_sharding import shard_dataset
from common.resnet import ResNet
from distributed_ml.simulation.send_grads.train import SendGradientsTrain
from distributed_ml.grad_processor.top_k_sparcifier import TopKSparcifier
from distributed_ml.grad_processor.k_means_quantizator import KMeansQuantizator, determine_size
import torch
import matplotlib.pyplot as plt


def main():
    train_dataset = load_cifar10(is_train=True, save_path='../../../data')
    test_dataset = load_cifar10(is_train=False, save_path='../../../data')
    train_shards = shard_dataset(train_dataset, shards_count=4, shuffle=True)
    model = ResNet(2)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == trainable_params
    print(f'Params number = {total_params}')

    simulator = SendGradientsTrain(
        model=model, epochs=10,
        opt_getter=lambda params: torch.optim.SGD(params, lr=0.1, weight_decay=0.0001, momentum=0.9),
        train_shards=train_shards, test_dataset=test_dataset,
        # gradient_processor=TopKSparcifier(k=[p.numel() // 5 for p in model.parameters()]),
        # gradient_processor=TopKSparcifier(k=total_params // 5),
        # gradient_processor=OneBitQuantizator(per_layer=True),
        # gradient_processor=OneBitQuantizator(per_layer=False),
        gradient_processor=KMeansQuantizator(size_determiner=determine_size),
        use_error_correction=True,
        train_batch_size=32,
        save_grad_dist=False
    )
    simulator.train()
    if simulator.grad_dist is not None:
        print(simulator.grad_dist)
        plt.figure()
        plt.hist(simulator.grad_dist)
        plt.savefig('../../../plots/grad_loss_none.png')


if __name__ == '__main__':
    main()
