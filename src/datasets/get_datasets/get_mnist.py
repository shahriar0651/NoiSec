import torchvision
import torchvision.transforms as transforms

def get_mnist_dataset(cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),])
    trainset = torchvision.datasets.MNIST(root=cfg.dataset_dir,
                                          transform=transform,
                                          train=True,
                                          download=True)
    testset = torchvision.datasets.MNIST(root=cfg.dataset_dir, 
                                         transform=transform,
                                         train=False,
                                         download=True)
    return trainset, testset