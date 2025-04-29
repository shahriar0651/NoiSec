import torchvision
import torchvision.transforms as transforms

def get_cifar100_dataset(cfg):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(10),  
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
  
    
    trainset = torchvision.datasets.CIFAR100(
        root=cfg.dataset_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(
        root=cfg.dataset_dir, train=False, download=True, transform=transform_test)
    return trainset, testset
