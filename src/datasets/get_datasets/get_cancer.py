import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

def get_cancer_dataset(cfg):
    ratio = 0.75
    train_data_dir = Path(f"{cfg.dataset_dir}/cancer/")
    if not train_data_dir.is_dir():
        print("Download Dataset!")
        # TODO: download_cancer_dataset(cfg)

    # Define the transforms for the training set
    train_transforms = transforms.Compose([
        transforms.Resize([150, 150]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness, contrast, saturation, and hue
        # Add normalization if needed
        # transforms.Normalize(mean=[0.6499, 0.4723, 0.5844], std=[0.1422, 0.1466, 0.1282])
    ])

    # Define the transforms for the validation set
    val_transforms = transforms.Compose([
        transforms.Resize([150, 150]),
        transforms.ToTensor(),
        # Add normalization if needed
        # transforms.Normalize(mean=[0.6499, 0.4723, 0.5844], std=[0.1422, 0.1466, 0.1282])
    ])

    # Load the full dataset without any transforms
    full_data = torchvision.datasets.ImageFolder(root=train_data_dir)

    # Split the dataset into training and validation sets
    n_train_examples = int(len(full_data) * ratio)
    n_val_examples = len(full_data) - n_train_examples
    train_indices, val_indices = torch.utils.data.random_split(range(len(full_data)), [n_train_examples, n_val_examples])

    # Create training and validation datasets with respective transforms
    trainset = torch.utils.data.Subset(full_data, train_indices)
    trainset.dataset.transform = train_transforms

    valset = torch.utils.data.Subset(full_data, val_indices)
    valset.dataset.transform = val_transforms

    return trainset, valset
