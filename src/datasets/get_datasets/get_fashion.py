import torchvision
import torchvision.transforms as transforms

def get_fashion_dataset(cfg):
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # Randomly rotate images within a range of 10 degrees
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        ])
    trainset = torchvision.datasets.FashionMNIST(root=cfg.dataset_dir, 
                                                 train=True,
                                                 download=True, 
                                                 transform=train_transform)
    testset = torchvision.datasets.FashionMNIST(root=cfg.dataset_dir,
                                                train=False,
                                                download=True, 
                                                transform=test_transform)
    return trainset, testset