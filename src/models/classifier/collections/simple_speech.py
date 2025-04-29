import torch
import torch.nn as nn
import torch.nn.functional as F

# class Simple_Classifier_SPEECH_v1(nn.Module):
#     def __init__(self, num_channels = 1, num_feats=128, num_classes = 35):
#         super(Simple_Classifier_SPEECH_v1, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc_feat = nn.Linear(64 * 8 * 10, num_feats)
#         self.fc_out = nn.Linear(num_feats, num_classes)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = self.pool(F.relu(self.conv4(x)))
#         x = x.view(-1, 64 * 8 * 10)
#         x = F.relu(self.fc_feat(x))
#         x = self.dropout(x)
#         x = self.fc_out(x)
#         return x


class Simple_Classifier_SPEECH_v1(nn.Module):
    def __init__(self, num_channels=1, num_feats=128, num_classes=35):
        super(Simple_Classifier_SPEECH_v1, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_feat = nn.Linear(64 * 4 * 5, num_feats)  # Corrected dimensions
        self.fc_out = nn.Linear(num_feats, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Output: (16, 32, 40)
        x = self.pool(F.relu(self.conv2(x)))  # Output: (32, 16, 20)
        x = self.pool(F.relu(self.conv3(x)))  # Output: (64, 8, 10)
        x = self.pool(F.relu(self.conv4(x)))  # Output: (64, 4, 5)
        x = x.view(-1, 64 * 4 * 5)
        x = F.relu(self.fc_feat(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        out = F.softmax(x, dim=1)
        # out = F.log_softmax(x, dim=1)
        return out

# Example usage:
# model = Simple_Classifier_SPEECH_v1()
# print(model)


class Simple_Classifier_SPEECH_v2(nn.Module):
    def __init__(self, num_channels = 1, num_feats=128, num_classes = 35):
        super(Simple_Classifier_SPEECH_v2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_feat = nn.Linear(64 * 8 * 10, num_feats)
        self.fc_out = nn.Linear(num_feats, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 10)
        x = F.relu(self.fc_feat(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        out = F.softmax(x, dim=1)
        # out = F.log_softmax(x, dim=1)
        return out
