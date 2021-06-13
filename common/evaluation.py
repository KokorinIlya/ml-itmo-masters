from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset
import torch


def calc_accuracy(model: torch.nn.Module, test_dataset: VisionDataset, batch_size: int = 128):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    total_correct = 0

    with torch.no_grad():
        for X, y in test_loader:
            y_hat = model(X).argmax(dim=1)
            total_correct += (y_hat == y).sum().item()

    return total_correct / len(test_dataset)
