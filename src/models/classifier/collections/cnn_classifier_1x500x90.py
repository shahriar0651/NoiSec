import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier_1x500x90_target(nn.Module):
    def __init__(self, num_channels=1, num_feats=512, num_classes=8):
        super(Classifier_1x500x90_target, self).__init__()
        KERNEL_SIZE = (3, 3)

        # Convolutional Blocks
        self.conv_layers = nn.Sequential(
            self._conv_block(num_channels, 32, KERNEL_SIZE),  # 1x500x90 -> 32x250x45
            self._conv_block(32, 64, KERNEL_SIZE),            # 32x250x45 -> 64x125x22
            self._conv_block(64, 64, KERNEL_SIZE),            # 64x125x22 -> 64x62x11
            self._conv_block(64, 128, KERNEL_SIZE),           # 64x62x11 -> 128x31x5
            self._conv_block(128, 128, KERNEL_SIZE),          # 128x31x5 -> 128x15x2
            self._conv_block(128, 256, KERNEL_SIZE),          # 128x15x2 -> 256x7x1
        )

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc_feat = nn.Linear(256 * 7 * 1, num_feats)  # Adjusted based on the flattened size
        self.dropout = nn.Dropout(0.25)
        self.fc_out = nn.Linear(num_feats, num_classes)
        
    def _conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc_feat(x))
        x = self.dropout(x)
        return self.fc_out(x)

    

class Classifier_1x500x90_surrogate(nn.Module):
    def __init__(self, num_channels=1, num_feats=512, num_classes=8):
        super(Classifier_1x500x90_surrogate, self).__init__()
        KERNEL_SIZE = (3, 3)

        # Convolutional Blocks
        self.conv_layers = nn.Sequential(
            self._conv_block(num_channels, 32, KERNEL_SIZE),  # 1x500x90 -> 32x250x45
            self._conv_block(32, 64, KERNEL_SIZE),            # 32x250x45 -> 64x125x22
            self._conv_block(64, 64, KERNEL_SIZE),            # 64x125x22 -> 64x62x11
            self._conv_block(64, 128, KERNEL_SIZE),           # 64x62x11 -> 128x31x5
            self._conv_block(128, 128, KERNEL_SIZE),          # 128x31x5 -> 128x15x2
            self._conv_block(128, 128, KERNEL_SIZE),          # 128x15x2 -> 256x7x1
        )

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc_feat = nn.Linear(128 * 7 * 1, num_feats)  # Adjusted based on the flattened size
        self.dropout = nn.Dropout(0.25)
        self.fc_out = nn.Linear(num_feats, num_classes)
        
    def _conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc_feat(x))
        x = self.dropout(x)
        return self.fc_out(x)
