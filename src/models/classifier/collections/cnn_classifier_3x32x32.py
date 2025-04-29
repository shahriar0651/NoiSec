
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier3x32x32_target(nn.Module):
    def __init__(self, num_channels = 3, num_feats=64, num_classes = 10):
        super(Classifier3x32x32_target, self).__init__()
        KERNEL_SIZE = (3, 3)
        # First Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=KERNEL_SIZE, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=KERNEL_SIZE, padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Second Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=KERNEL_SIZE, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=KERNEL_SIZE, padding='same')
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Third Convolutional Block
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=KERNEL_SIZE, padding='same')
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=KERNEL_SIZE, padding='same')
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.25)
        
        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc_feat = nn.Linear(2048, num_feats)  # Assuming 3 pooling layers with stride=2
        self.dropout4 = nn.Dropout(0.25)
        self.fc_out = nn.Linear(num_feats, num_classes)
        
    def forward(self, x):
        # First Convolutional Block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second Convolutional Block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third Convolutional Block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Fully Connected Layers
        x = self.flatten(x)
        x = F.relu(self.fc_feat(x))
        x = self.dropout4(x)
        x = self.fc_out(x)
        # x = torch.sigmoid(x)
        return x

class Classifier3x32x32_surrogate(nn.Module):
    def __init__(self, num_channels = 3, num_feats=64, num_classes = 10):
        super(Classifier3x32x32_surrogate, self).__init__()
        KERNEL_SIZE = (3, 3)
        # First Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=KERNEL_SIZE, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=KERNEL_SIZE, padding='same')
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Second Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=KERNEL_SIZE, padding='same')
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=KERNEL_SIZE, padding='same')
        self.bn4 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        
        # # Third Convolutional Block
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=KERNEL_SIZE, padding='same')
        # self.bn5 = nn.BatchNorm2d(128)
        # self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=KERNEL_SIZE, padding='same')
        # self.bn6 = nn.BatchNorm2d(128)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout3 = nn.Dropout(0.25)
        
        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc_feat = nn.Linear(2048, num_feats)  # Assuming 3 pooling layers with stride=2
        self.dropout4 = nn.Dropout(0.25)
        self.fc_out = nn.Linear(num_feats, num_classes)
        
    def forward(self, x):
        # First Convolutional Block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second Convolutional Block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # # Third Convolutional Block
        # x = F.relu(self.bn5(self.conv5(x)))
        # x = F.relu(self.bn6(self.conv6(x)))
        # x = self.pool3(x)
        # x = self.dropout3(x)
        
        # Fully Connected Layers
        x = self.flatten(x)
        x = F.relu(self.fc_feat(x))
        x = self.dropout4(x)
        x = self.fc_out(x)
        # x = torch.sigmoid(x)
        return x
