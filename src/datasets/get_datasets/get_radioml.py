"""
Collcted from:
https://github.com/isaaccorley/pytorch-modulation-recognition/blob/master/torch_modulation_recognition/data.py#L32
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

def get_radioml_dataset(cfg):
    # List of classes
    version = f'{cfg.dataset.version}.dat'
    classes = cfg.dataset.classes
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    # Function to get the index of a class
    def get_index(class_name):
        return class_to_index.get(class_name, "Class not found")


    MODULATIONS = {mod : indx for indx, mod in enumerate(classes)}
    SNRS = cfg.dataset.snrs
        
    # Signal-to-Noise Ratios
    

    class RadioML2016(torch.utils.data.Dataset):

        URL = "http://opendata.deepsig.io/datasets/2016.10/RML2016.10b.tar.bz2"
        modulations = MODULATIONS
        snrs = SNRS

        def __init__(
            self,
            data_dir: str = f'{cfg.dataset_dir}/RadioML',
            file_name: str = f"{cfg.dataset.version}.dat"
        ):
            self.file_name = file_name
            self.data_dir = data_dir
            self.n_classes = cfg.dataset.num_classes
            self.X, self.y = self.load_data()
            gc.collect()

        def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
            """ Load data from file """
            print("Loading dataset from file...")
            with open(os.path.join(self.data_dir, self.file_name), "rb") as f:
                data = pickle.load(f, encoding="latin1")
            
            X, y = [], []
            print("Processing dataset")
            for mod, snr in tqdm(list(itertools.product(self.modulations, self.snrs))):
                X.append(data[(mod, snr)])

                for i in range(data[(mod, snr)].shape[0]):
                    y.append((mod, snr))

            X = np.vstack(X)
            
            return X, y

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            """ Load a batch of input and labels """
            x, (mod, snr) = self.X[idx], self.y[idx]
            
            # TODO: Scale the data from 0 to 1
            x_min = np.min(x) 
            x_max = np.max(x) 
            x = (x - x_min) / (x_max - x_min)

            y = self.modulations[mod]
            x, y = torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
            x = x.to(torch.float).unsqueeze(0)
            return x, y

        def __len__(self) -> int:
            return self.X.shape[0]

        def get_signals(self, mod: List[str] = None, snr: List[int] = None) -> Dict:
            """ Return signals of a certain modulation or signal-to-noise ratio """

            # If None then it means all mods or snrs
            if mod is None:
                modulations = self.modulations.copy()
            if snr is None:
                snrs = self.snrs.copy()

            # If single mod or snr then convert to list to make iterable
            if not isinstance(mod, List):
                modulations = [mod]
            if not isinstance(snr, List):
                snrs = [snr]

            # Aggregate signals into a dictionary
            X = {}
            for mod, snr in list(itertools.product(modulations, snrs)):
                X[(mod, snr)] = []
                for idx, (m, s) in enumerate(self.y):
                    if m == mod and s == snr:
                        X[(mod, snr)].append(np.expand_dims(self.X[idx, ...], axis=0))
                
                X[(mod, snr)] =  np.concatenate(X[(mod, snr)], axis=0)

            return X
        

    # Params

    dataset = RadioML2016()
    # Split into train/val sets
    total = len(dataset)
    lengths = [int(len(dataset)*0.75)]
    lengths.append(total - lengths[0])
    print("Splitting into {} train and {} val".format(lengths[0], lengths[1]))
    trainset, testset = random_split(dataset, lengths)
    return trainset, testset
