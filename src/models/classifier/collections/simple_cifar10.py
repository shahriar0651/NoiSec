import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple_Classifier_CIFAR10_target(nn.Module):
    def __init__(self):
        super(Simple_Classifier_CIFAR10_target, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 256)
        self.fc_feat = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.relu(F.max_pool2d(self.conv4(x),2))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        feat = F.relu(self.fc_feat(x))

        # out = F.softmax(self.fc_out(feat), dim=1)
        out = F.log_softmax(self.fc_out(feat), dim=1)

        # return F.log_softmax(x, dim=1)
        # out = F.softmax(x, dim=1)
        return out


class Simple_Classifier_CIFAR10_surrogate(nn.Module):
    def __init__(self):
        super(Simple_Classifier_CIFAR10_surrogate, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc_feat = nn.Linear(512, 64)
        self.fc_out = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Output: (batch_size, 32, 30, 30)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Output: (batch_size, 32, 14, 14)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))  # Output: (batch_size, 64, 6, 6)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 64 * 6 * 6)
        x = F.relu(self.fc1(x))  # Output: (batch_size, 512)
        feat = F.relu(self.fc_feat(x))  # Output: (batch_size, 64)
        out = F.log_softmax(self.fc_out(feat), dim=1)  # Output: (batch_size, 10)
        return out

    
