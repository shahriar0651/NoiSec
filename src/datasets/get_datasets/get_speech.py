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

def get_speech_dataset(cfg):

    # Define the preprocessing transform
    transform = nn.Sequential(
        MelSpectrogram(sample_rate=16000, n_mels=64),
        AmplitudeToDB()
    )

    # List of classes
    classes = cfg.dataset.classes
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    # Function to get the index of a class
    def get_index(class_name):
        return class_to_index.get(class_name, "Class not found")

    class SpeechCommandsDataset(SPEECHCOMMANDS):
        def __init__(self, subset: str = None):
            super().__init__(cfg.dataset_dir, download=True)
            self.subset = subset
            if self.subset == "validation":
                self._walker = self._load_list("validation_list.txt")
            elif self.subset == "testing":
                self._walker = self._load_list("testing_list.txt")
            elif self.subset == "training":
                excludes = set(self._load_list("validation_list.txt") + self._load_list("testing_list.txt"))
                self._walker = [w for w in self._walker if w not in excludes]

        def _load_list(self, filename):
            with open(os.path.join(self._path, filename)) as f:
                return [os.path.join(self._path, line.strip()) for line in f]

        def __getitem__(self, n: int):
            waveform, sample_rate, label, *_ = super().__getitem__(n)
            # Apply the transformation to the waveform

            if waveform.size(1) < 16000:
                waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.size(1)))
            else:
                waveform = waveform[:, :16000]

            if transform is not None:
                spectrum = transform(waveform)
            else:
                spectrum = waveform
            # print(waveform.shape, spectrum.shape)
            # Scale the spectrum to the range 0 to 1
            spectrum_min = spectrum.min()
            spectrum_max = spectrum.max()
            spectrum = (spectrum - spectrum_min) / (spectrum_max - spectrum_min)
        
            label = get_index(label)
            return spectrum, label

    # Fit the label encoder on the training dataset
    trainset = SpeechCommandsDataset(subset="training")
    testset = SpeechCommandsDataset(subset="testing")
    return trainset, testset
