import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common.evaluation import calc_accuracy


def train_model(model, epochs,
                train_dataset, test_dataset,
                teacher=None, alpha=0.5,
                train_batch_size=128, test_batch_size=128,
                epochs_passed=0):
    if teacher is not None:
        teacher.eval()

    opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001, momentum=0.9)
    train_start_time = time.time()
    accs = []
    acc = calc_accuracy(model, test_dataset, batch_size=test_batch_size)
    if epochs_passed == 0:
        accs.append(acc)
    print("Initial acc = {0}".format(acc))

    for epoch in range(epochs):
        model.train()
        epoch_start_time = time.time()

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        for X, y in train_loader:
            opt.zero_grad()
            y_hat = model(X)

            target_loss = F.cross_entropy(y_hat, y)
            if teacher is None:
                total_loss = target_loss
            else:
                y_hat_teacher = teacher(X)
                dist_loss = F.mse_loss(y_hat, y_hat_teacher)
                total_loss = alpha * target_loss + (1 - alpha) * dist_loss

            total_loss.backward()
            opt.step()

        acc = calc_accuracy(model, test_dataset, batch_size=test_batch_size)
        accs.append(acc)

        cur_time = time.time()
        epoch_time_spent = int(cur_time - epoch_start_time)
        total_time_spent = int(cur_time - train_start_time)
        print(
            "Epochs passed = {0}, acc = {1}, seconds per epoch = {2}, total seconds elapsed = {3}".format(
                epochs_passed + epoch + 1, acc, epoch_time_spent, total_time_spent
            )
        )

    return accs
