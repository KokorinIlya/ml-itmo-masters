import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common.evaluation import calc_accuracy


def __get_grad_single_shard(model, train_shard_iter):
    model.zero_grad()
    try:
        X, y = next(train_shard_iter)
        y_hat = model(X)
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        return [x.grad.detach() for x in model.parameters()], True, X, y
    except StopIteration:
        return [torch.zeros_like(x) for x in model.parameters()], False, None, None


def __collect_grads(model, train_iters):
    grads = []
    has_grads = False
    X, y = None, None
    for shard_iter in train_iters:
        cur_grad, has_grad, X_cur, y_cur = __get_grad_single_shard(model, shard_iter)

        if X is None:
            X = X_cur
            y = y_cur
        elif X_cur is not None:
            X = torch.vstack((X, X_cur))
            y = torch.hstack((y, y_cur))

        grads.append(cur_grad)
        has_grads = has_grads or has_grad
    return grads, has_grads, X, y


def __do_step(model, opt, grads, shards_count, true_grad):
    model.zero_grad()
    params_list = list(model.parameters())
    for cur_param_grads in grads:
        assert len(params_list) == len(cur_param_grads)
    for cur_param, cur_true_grad, *cur_param_grads in zip(model.parameters(), true_grad, *grads):
        assert len(cur_param_grads) == shards_count
        result_grad = torch.zeros_like(cur_param_grads[0])
        for cur_param_grad in cur_param_grads:
            assert result_grad.size() == cur_param_grad.size()
            result_grad += cur_param_grad
        assert cur_param.size() == result_grad.size()
        result_grad /= shards_count
        assert ((cur_true_grad - result_grad).abs() < 1e-4).all().item()
        cur_param.grad = result_grad
    opt.step()


def __calc_grad(model, X, y):
    model.zero_grad()
    y_hat = model(X)
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    return [x.grad.detach() for x in model.parameters()]


def train_distributed(model, epochs, lr,
                      train_shards, test_dataset,
                      train_batch_size=128, test_batch_size=128):
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9)
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
            grads, has_grads, X, y = __collect_grads(model, train_iters)
            if has_grads:
                assert X is not None and y is not None
                true_grad = __calc_grad(model, X, y)
                __do_step(model=model, opt=opt, grads=grads, shards_count=len(train_shards), true_grad=true_grad)
            else:
                break

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

    return accs
