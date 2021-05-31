from torchvision import transforms
import torchvision


def load_cifar10(is_train: bool, save_path: str) -> torchvision.datasets.CIFAR10:
    if is_train:
        transformers = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=(32, 32), padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    else:
        transformers = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

    return torchvision.datasets.CIFAR10(
        root=save_path, train=is_train, download=True,
        transform=transforms.Compose(transformers)
    )
