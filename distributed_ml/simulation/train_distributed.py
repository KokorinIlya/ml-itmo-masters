import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common.evaluation import calc_accuracy


def __get_grad_single_shard(model, train_shard_iter):
    try:
        model.zero_grad()
        X, y = next(train_shard_iter)
        y_hat = model(X)
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        return [x.grad.detach() for x in model.parameters()], True
    except StopIteration:
        return [torch.zeros_like(x) for x in model.parameters()], False


def __collect_grads(model, train_iters):
    grads = []
    has_grads = False
    for shard_iter in train_iters:
        cur_grad, has_grad = __get_grad_single_shard(model, shard_iter)
        grads.append(cur_grad)
        has_grads = has_grads or has_grad
    return grads, has_grads


def __do_step(model, opt, grads, shards_count):
    for cur_param, *cur_param_grads in zip(model.parameters(), *grads):
        assert len(cur_param_grads) == shards_count
        result_grad = torch.zeros_like(cur_param_grads[0])
        for cur_param_grad in cur_param_grads:
            assert result_grad.size() == cur_param_grad.size()
            result_grad += cur_param_grad
        cur_param.grad = result_grad
    opt.step()


def train_distributed(model, epochs,
                      train_shards, test_dataset,
                      train_batch_size=128, test_batch_size=128):
    opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001, momentum=0.9)
    train_start_time = time.time()
    accs = []
    acc = calc_accuracy(model, test_dataset, batch_size=test_batch_size)
    print("Initial acc = {0}".format(acc))

    for epoch in range(epochs):
        model.train()
        epoch_start_time = time.time()

        train_iters = [
            iter(DataLoader(train_shard, batch_size=train_batch_size, shuffle=True))
            for train_shard in train_shards
        ]

        while True:
            grads, has_grads = __collect_grads(model, train_iters)
            if has_grads:
                __do_step(model=model, opt=opt, grads=grads, shards_count=len(train_shards))
                continue
            else:
                acc = calc_accuracy(model, test_dataset, batch_size=test_batch_size)
                accs.append(acc)

                cur_time = time.time()
                epoch_time_spent = int(cur_time - epoch_start_time)
                total_time_spent = int(cur_time - train_start_time)
                print(
                    "Epochs passed = {0}, acc = {1}, seconds per epoch = {2}, total seconds elapsed = {3}".format(
                        epoch + 1, acc, epoch_time_spent, total_time_spent
                    )
                )
                break

    return accs
