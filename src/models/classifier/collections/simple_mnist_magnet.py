import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple_Classifier_MNIST_magnet(nn.Module):
    def __init__(self):
        super(Simple_Classifier_MNIST_magnet, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 200)
        self.fc_feat = nn.Linear(200, 64)
        self.fc_out = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x),2)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, 0.50)
        feat = F.relu(self.fc_feat(x))
        # out = F.softmax(self.fc_out(feat), dim=1)
        out = self.fc_out(feat)
        return out
