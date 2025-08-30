
import torch
import os
import gc
import torch
import numpy as np
from typing import Tuple
from torch.utils.data import random_split
import requests
from pathlib import Path


def download_zenodo(record_id, target_dir, renamed_folder):
    """
    Simple function to download all files from a Zenodo record into a folder
    and rename the folder afterwards.
    """
    # Create target directory
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Get file list from Zenodo API
    api_url = f"https://zenodo.org/api/records/{record_id}"
    data = requests.get(api_url).json()

    for f in data["files"]:
        file_url = f["links"]["self"]
        file_name = f["key"]
        file_path = target_path / file_name

        print(f"Downloading {file_name}...")
        with requests.get(file_url, stream=True) as r:
            with open(file_path, "wb") as out:
                for chunk in r.iter_content(chunk_size=8192):
                    out.write(chunk)

    # Rename folder
    new_path = target_path.parent / renamed_folder
    target_path.rename(new_path)
    print(f"Done. Files saved in: {new_path}")
    # Example usage:
    # download_zenodo("16995272", "zenodo_files", "my_dataset")

class ActivityDataset(torch.utils.data.Dataset):
    """
    Original dataset: https://github.com/ludlows/CSI-Activity-Recognition/tree/master
    """

    def __init__(
        self,
        data_dir: str,
        n_classes : list,
    ):
        self.data_dir = data_dir
        self.n_classes = n_classes
        if not Path(self.data_dir).exists():
            print("The dataset does not exist! Downloading from zenodo...")
            download_zenodo("16995272", data_dir, "CSIActivity")
        self.X_all = np.load(f'{self.data_dir}/X_all.npy')
        self.Y_all = np.argmax(np.load(f'{self.data_dir}/Y_all.npy'), axis=1)
        gc.collect()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Load a batch of input and labels """
        x, y = self.X_all[idx], self.Y_all[idx]
        x_min = np.min(x) 
        x_max = np.max(x) 
        x = (x - x_min) / (x_max - x_min)
        x, y = torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
        x = x.to(torch.float).unsqueeze(0)
        return x, y

    def __len__(self) -> int:
        return self.X_all.shape[0]


def get_activity_dataset(cfg):
    # List of classes
    classes = cfg.dataset.classes
    data_dir = f'{cfg.dataset_dir}/CSIActivity'
    dataset = ActivityDataset(data_dir, classes)
    total = len(dataset)
    lengths = [int(len(dataset)*0.75)]
    lengths.append(total - lengths[0])
    print("Splitting into {} train and {} val".format(lengths[0], lengths[1]))
    trainset, testset = random_split(dataset, lengths)
    return trainset, testset
