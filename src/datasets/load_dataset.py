#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import copy
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchaudio.datasets import SPEECHCOMMANDS
import os
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from .get_datasets.get_mnist import get_mnist_dataset
from .get_datasets.get_cifar10 import get_cifar10_dataset
from .get_datasets.get_fashion import get_fashion_dataset
from .get_datasets.get_gtsrb import get_gtsrb_dataset
from .get_datasets.get_cancer import get_cancer_dataset
from .get_datasets.get_speech import get_speech_dataset
from .get_datasets.get_cifar100 import get_cifar100_dataset
from .get_datasets.get_radioml import get_radioml_dataset
from .get_datasets.get_robofi import get_robofi_dataset
from .get_datasets.get_activity import get_activity_dataset
from .get_datasets.get_medmnist import get_medmnist_dataset
from helper import *

# functions to show an image


def data_loader(cfg, model_type):
    
    batch_size = cfg.dataset.batch_size
    if cfg.dataset.name == 'mnist':
        trainset, testset =  get_mnist_dataset(cfg)
    elif cfg.dataset.name == 'cifar10':
        trainset, testset =  get_cifar10_dataset(cfg)
    elif cfg.dataset.name == 'fashion':
        trainset, testset =  get_fashion_dataset(cfg)
    elif cfg.dataset.name == 'gtsrb':
        trainset, testset =  get_gtsrb_dataset(cfg) 
    elif cfg.dataset.name == 'cancer':
        trainset, testset =  get_cancer_dataset(cfg)     
    elif cfg.dataset.name == 'speech':
        trainset, testset =  get_speech_dataset(cfg) 
    elif cfg.dataset.name == 'cifar100':
        trainset, testset =  get_cifar100_dataset(cfg) 
    elif cfg.dataset.name in ['radiomlv1', 'radiomlv2', 'radiomlv3']:
        trainset, testset =  get_radioml_dataset(cfg) 
    elif cfg.dataset.name == 'robofi':
        trainset, testset =  get_robofi_dataset(cfg) 
    elif cfg.dataset.name == 'activity':
        trainset, testset =  get_activity_dataset(cfg) 
    elif cfg.dataset.name == 'medmnist':
        trainset, testset =  get_medmnist_dataset(cfg) 
    else:
        print("Dataset not implemented yet!")

    poison_ratio = cfg.dataset.poison_ratio if model_type=='badnet' else 0.0
    max_val = 255 if model_type=='badnet' else 1

    # Determine the splitting index
    split_index = int(len(testset) * 0.75)

    # Split the dataset
    testset_1 = Subset(testset, range(0, split_index))
    testset_2 = Subset(testset, range(split_index, len(testset)))

    print(f"Length of testset_1: {len(testset_1)}")
    print(f"Length of testset_2: {len(testset_2)}")

    trigger_label=cfg.dataset.atk_cfg["BadNet"]["trigger_label"]
    trainset_trg = PoisonedDataset(cfg,
                                   dataset=trainset, 
                                    dataname=cfg.dataset.name,
                                    trigger_label=trigger_label,
                                    portion=poison_ratio,
                                    mode="train",
                                    device=cfg.device,
                                    max_val = max_val, 
                                    )
    testset_cln = PoisonedDataset(cfg,
                                  dataset=testset_1, 
                                    dataname=cfg.dataset.name,
                                    trigger_label=trigger_label,
                                    portion=0.0,
                                    mode="test",
                                    device=cfg.device,
                                    max_val = max_val) 
    
    testset_val = PoisonedDataset(cfg,
                                  dataset=testset_2, 
                                    dataname=cfg.dataset.name,
                                    trigger_label=trigger_label,
                                    portion=0.0,
                                    mode="test",
                                    device=cfg.device,
                                    max_val = max_val) 
    
    testset_trg = PoisonedDataset(cfg,
                                  dataset=testset_1, 
                                    dataname=cfg.dataset.name,
                                    trigger_label=trigger_label,
                                    portion=1.0,
                                    mode="test",
                                    device=cfg.device,
                                    max_val = max_val
                                    )

    min_value = float('inf')
    max_value = float('-inf')
    print(len(trainset_trg))
    for index in range(len(trainset_trg)):
        image, label = trainset_trg[index]
        # print(image, label)
        min_value = min(min_value, torch.min(image))
        max_value = max(max_value, torch.max(image))
    # # Print the result    
    # print("-------------------------------")
    # print("After the Class: ")
    # print("Min Value:", min_value.item())
    # print("Max Value:", max_value.item())
    # print("Shape of the labels: ", label.shape)
    # print("-------------------------------\n\n")
    # if cfg.dataset.name != 'speech':

    # Set the random seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)

    train_loader_trg = torch.utils.data.DataLoader(trainset_trg, 
                                                batch_size=batch_size,
                                                shuffle=True)
    test_loader_cln = torch.utils.data.DataLoader(testset_cln, 
                                                batch_size=batch_size,
                                                shuffle=False)
    test_loader_val = torch.utils.data.DataLoader(testset_val, 
                                                batch_size=batch_size,
                                                shuffle=False)
    test_loader_trg = torch.utils.data.DataLoader(testset_trg, 
                                                batch_size=batch_size,
                                                shuffle=False)        
       
    # get some random training images
    dataiter = iter(train_loader_trg)
    images, labels = next(dataiter)
    imshow(cfg, torchvision.utils.make_grid(images, padding = 5, pad_value= 1.0,), model_type)
    return train_loader_trg, test_loader_cln, test_loader_val, test_loader_trg




