import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# class Simple_Classifier_CANCER_target(nn.Module):
#     def __init__(self):
#         super(Simple_Classifier_CANCER_target, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
#         self.fc1 = nn.Linear(64 * 16 * 16, 512)  # Update this line
#         self.fc_feat = nn.Linear(512, 128)
#         self.fc_out = nn.Linear(128, 8)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.relu(F.max_pool2d(self.conv3(x), 2))
#         x = F.relu(F.max_pool2d(self.conv4(x), 2))
#         x = x.view(-1, 64 * 16 * 16)  # Update this line
#         x = F.relu(self.fc1(x))
#         feat = F.relu(self.fc_feat(x))
#         out = F.softmax(self.fc_out(feat), dim=1)
#         return out



class Simple_Classifier_CANCER_target(nn.Module):
    def __init__(self):
        super(Simple_Classifier_CANCER_target, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 9 * 9, 512)  # Adjust the input size according to the output of the conv layers
        self.fc_feat = nn.Linear(512, 128)
        self.fc_out = nn.Linear(128, 8)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 75x75x32
        x = self.pool(F.relu(self.conv2(x)))  # 37x37x64
        x = self.pool(F.relu(self.conv3(x)))  # 18x18x128
        x = self.pool(F.relu(self.conv4(x)))  # 9x9x256
        
        # Flatten the tensor
        x = x.view(-1, 256 * 9 * 9)  # Flattened output
        
        # Fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc_feat(x))
        x = self.dropout(x)
        
        # Output layer with softmax
        x = F.softmax(self.fc_out(x), dim=1)
        
        return x

class Simple_Classifier_CANCER_surrogate(nn.Module):
    def __init__(self):
        super(Simple_Classifier_CANCER_surrogate, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 35 * 35, 512)  # Update this line
        self.fc_feat = nn.Linear(512, 128)
        self.fc_out = nn.Linear(128, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 64 * 35 * 35)  # Update this line
        x = F.relu(self.fc1(x))
        feat = F.relu(self.fc_feat(x))
        out = F.softmax(self.fc_out(feat), dim=1)
        return out