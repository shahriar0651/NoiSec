#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torchvision
import torchvision.transforms as transforms

def get_cifar10_dataset(cfg):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Randomly crop the image with padding
        transforms.RandomHorizontalFlip(),     # Randomly flip the image horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness, contrast, saturation, and hue
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Width and height shift
        transforms.RandomRotation(15),         # Randomly rotate the image by up to 15 degrees
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root=cfg.dataset_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=cfg.dataset_dir, train=False, download=True, transform=transform_test)
    return trainset, testset
