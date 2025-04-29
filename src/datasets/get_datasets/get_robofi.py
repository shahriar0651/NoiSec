"""
Collcted from:
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


def get_robofi_dataset(cfg):
    # List of classes
    classes = cfg.dataset.classes
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    class RoboFiSense(torch.utils.data.Dataset):

        URL = "https://github.com/SiamiLab/RoboFiSense"
        modulations = class_to_index

        def __init__(
            self,
            data_dir: str = f'{cfg.dataset_dir}/RoboFiSense',
        ):
            self.data_dir = data_dir
            self.n_classes = cfg.dataset.num_classes
            self.X_all, self.Y_all = self.load_data()
            gc.collect()

        def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
            CSIs = []
            Labels = []
            directory = self.data_dir
            for location in os.listdir(directory):
                location_path = os.path.join(directory, location)
                # print("Location: ", location)
                for labels in os.listdir(location_path):  # Main Directory where each class label is present as a folder name.
                    # print("labels: ", labels)
                    label_path = os.path.join(location_path, labels)
                    if os.path.isdir(label_path):
                        if labels == 'Arc':  # Folder contains ARCXZ CSIs get the '0' class label.
                            label = 0
                        elif labels == 'Elbow':
                            label = 1
                        elif labels == 'Rectangle':
                            label = 2
                        elif labels == 'Silence':
                            label = 3  
                        elif labels == 'SLFW':
                            label = 4
                        elif labels == 'SLRL':
                            label = 5
                        elif labels == 'SLUD':
                            label = 6
                        elif labels == 'Triangle':
                            label = 7
                        else:
                            print(f"label {labels} does not exist")
                            return 0

                        # Get a list of speeds in the label directory
                        speed_folders = os.listdir(label_path)
                        for speed_folder in speed_folders:
                            # print("speed_folders ", speed_folders)

                            speed_path = os.path.join(label_path, speed_folder)
                            for csi_file in os.listdir(speed_path):  # Extracting the file name of the csi from Speed folder
                                # print("csi_file :", csi_file)
                                if '.cmplx' not in csi_file: 
                                    continue  # Skip processing this file
                                def read_file(filename):
                                    with open(filename, 'rb') as FID:
                                        mp = pickle.Unpickler(FID)
                                        data = mp.load()
                                    return data

                                csi = read_file(os.path.join(speed_path, csi_file))  # Reading the csi
                                ar1 = np.asmatrix(np.abs(csi[0]['complex_csi']))
                                arr1 = np.delete(ar1, [0, 1, 2, 3, 4, 115, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 253, 254, 255], 1)
                                arr1 = np.pad(arr1, ((0, 0), (64, 64)), mode='constant')

                                # ar2 = np.asmatrix(np.abs(csi[1]['complex_csi']))
                                # arr2 = np.delete(ar2, [0, 1, 2, 3, 4, 115, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 253, 254, 255], 1)
                                # arr2 = np.pad(arr2, ((0, 0), (64, 64)), mode='constant')

                                # arr = np.concatenate((arr1, arr2), axis=1)
                                arr = arr1

                                CSIs.append(arr.T)
                                Labels.append(label)
                            
            X = np.array(CSIs)
            y = np.vstack(Labels)

            print(X.shape)
            print(y.shape)

            return X, y             

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


    dataset = RoboFiSense()
    total = len(dataset)
    lengths = [int(len(dataset)*0.75)]
    lengths.append(total - lengths[0])
    print("Splitting into {} train and {} val".format(lengths[0], lengths[1]))
    trainset, testset = random_split(dataset, lengths)
    return trainset, testset
