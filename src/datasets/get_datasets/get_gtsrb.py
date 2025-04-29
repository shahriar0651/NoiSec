#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path


def get_gtsrb_dataset(cfg):

    # torchvision.datasets.GTSRB(
    transform_train = transforms.Compose([
        transforms.Resize(36),              # Resize to 256x256
        transforms.RandomCrop(32, padding=4),  # Randomly crop the image with padding
        # transforms.RandomHorizontalFlip(),     # Randomly flip the image horizontally
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness, contrast, saturation, and hue
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Width and height shift
        # transforms.RandomRotation(15),         # Randomly rotate the image by up to 15 degrees
        transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.Resize(36),              # Resize to 256x256
        transforms.CenterCrop(32),          # Crop center to 224x224      
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    trainset = torchvision.datasets.GTSRB(root = cfg.dataset_dir, split='train',
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.GTSRB(root=cfg.dataset_dir, split='test',
                                        download=True, transform=transform_test)   

    return trainset, testset


