import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier1x64x81_target(nn.Module):
    def __init__(self, num_channels=1, num_feats=256, num_classes=35):
        super(Classifier1x64x81_target, self).__init__()
        KERNEL_SIZE = (3, 3)

        # Convolutional Blocks
        self.conv_layers = nn.Sequential(
            self._conv_block(num_channels, 32, KERNEL_SIZE),
            self._conv_block(32, 64, KERNEL_SIZE),
            self._conv_block(64, 128, KERNEL_SIZE),
            self._conv_block(128, 256, KERNEL_SIZE),
            self._conv_block(256, 512, KERNEL_SIZE)
        )

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc_feat = nn.Linear(2048, num_feats)
        self.dropout = nn.Dropout(0.25)
        self.fc_out = nn.Linear(num_feats, num_classes)
        
    def _conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = F.relu(self.fc_feat(x))
        x = self.dropout(x)
        return self.fc_out(x)


class Classifier1x64x81_surrogate(nn.Module):
    def __init__(self, num_channels=1, num_feats=256, num_classes=35):
        super(Classifier1x64x81_surrogate, self).__init__()
        KERNEL_SIZE = (3, 3)

        # Convolutional Blocks
        self.conv_layers = nn.Sequential(
            self._conv_block(num_channels, 32, KERNEL_SIZE),
            self._conv_block(32, 64, KERNEL_SIZE),
            self._conv_block(64, 64, KERNEL_SIZE),
            self._conv_block(64, 128, KERNEL_SIZE),
            self._conv_block(128, 128, KERNEL_SIZE)
        )

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc_feat = nn.Linear(512, num_feats)
        self.dropout = nn.Dropout(0.25)
        self.fc_out = nn.Linear(num_feats, num_classes)
        
    def _conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = F.relu(self.fc_feat(x))
        x = self.dropout(x)
        return self.fc_out(x)
