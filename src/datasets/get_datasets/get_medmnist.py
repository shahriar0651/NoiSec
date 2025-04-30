# https://github.com/MedMNIST/MedMNIST/blob/main/examples/getting_started.ipynb

import medmnist
from medmnist import INFO
import torchvision.transforms as transforms
from collections import Counter

def get_medmnist_dataset(cfg):
    data_flag = 'pneumoniamnist' #cfg.dataset.name
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])


    transform = transforms.Compose([
        transforms.ToTensor(),])
    
    trainset = DataClass(root=cfg.dataset_dir,
                         transform=transform,
                         split='train',
                         download=True,
                         size=cfg.dataset.width)
    testset = DataClass(root=cfg.dataset_dir, 
                        transform=transform,
                        split='test',
                        download=True,
                        size=cfg.dataset.width)


    return trainset, testset