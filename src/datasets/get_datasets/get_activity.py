"""
Collcted and nodified from:
https://github.com/SiamiLab/RoboFiSense
"""


import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchaudio.datasets import SPEECHCOMMANDS
import os
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

import torch
import torchaudio.transforms as T
from torchaudio.datasets import SPEECHCOMMANDS
from sklearn.preprocessing import LabelEncoder

import os
import gc
import itertools
import pickle
import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict
from torch.utils.data import DataLoader, random_split


def get_activity_dataset(cfg):
    # List of classes
    classes = cfg.dataset.classes
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    class ActivityDataset(torch.utils.data.Dataset):

        URL = "https://github.com/ludlows/CSI-Activity-Recognition/tree/master"

        def __init__(
            self,
            data_dir: str = f'{cfg.dataset_dir}/CSIActivity',
        ):
            self.data_dir = data_dir
            self.n_classes = cfg.dataset.num_classes
            self.X_all = np.load(f'{self.data_dir}/X_all.npy')
            self.Y_all = np.argmax(np.load(f'{self.data_dir}/Y_all.npy'), axis=1)
            gc.collect()

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            """ Load a batch of input and labels """
            x, y = self.X_all[idx], self.Y_all[idx]
            
            # TODO: Scale the data from 0 to 1
            x_min = np.min(x) 
            x_max = np.max(x) 
            x = (x - x_min) / (x_max - x_min)
            x, y = torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
            x = x.to(torch.float).unsqueeze(0)
            return x, y

        def __len__(self) -> int:
            return self.X_all.shape[0]


    dataset = ActivityDataset()
    total = len(dataset)
    lengths = [int(len(dataset)*0.75)]
    lengths.append(total - lengths[0])
    print("Splitting into {} train and {} val".format(lengths[0], lengths[1]))
    trainset, testset = random_split(dataset, lengths)
    return trainset, testset
