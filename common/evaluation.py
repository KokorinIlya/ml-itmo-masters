from torch.utils.data import DataLoader


def calc_accuracy(model, test_dataset, batch_size=128):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    total_correct = 0

    for X, y in test_loader:
        y_hat = model(X).argmax(dim=1)
        total_correct += (y_hat == y).sum().item()

    return total_correct / len(test_dataset)
