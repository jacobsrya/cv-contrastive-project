# src/datasets.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class ContrastiveTransformCIFAR10:
    """Return TWO different augmented views of the same image."""

    def __init__(self):
        normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

        self.transform_view1 = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.6, 1.0)),  # geometric
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(                         # photometric
                brightness=0.4, contrast=0.4,
                saturation=0.4, hue=0.1
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # blur
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(                    # occlusion
                p=0.2, scale=(0.02, 0.15),
                ratio=(0.3, 3.3), value=0, inplace=False
            ),
        ])

        self.transform_view2 = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.05
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(
                p=0.2, scale=(0.02, 0.15),
                ratio=(0.3, 3.3), value=0, inplace=False
            ),
        ])

    def __call__(self, x):
        v1 = self.transform_view1(x)
        v2 = self.transform_view2(x)
        return v1, v2


def get_cifar10_contrastive_dataloader(data_dir: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    transform = ContrastiveTransformCIFAR10()

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,  # returns (view1, view2)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader


# add near top if not already there:
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

def get_cifar10_eval_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
):
    """
    Returns train/test DataLoaders for downstream evaluation.
    These use LIGHT transforms: ToTensor + Normalize only.
    """
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=eval_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
