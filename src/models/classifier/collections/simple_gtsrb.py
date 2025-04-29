import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple_Classifier_GTSRB_target(nn.Module):
    def __init__(self):
        super(Simple_Classifier_GTSRB_target, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 256)
        self.fc_feat = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 43)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.relu(F.max_pool2d(self.conv4(x),2))
        # print("Shape before flattening: ", x.shape)
        x = x.view(-1, 256)
        # x = F.relu(self.fc1(x))
        feat = F.relu(self.fc_feat(x))
        # out = F.softmax(self.fc_out(feat), dim=1)
        out = F.softmax(self.fc_out(feat), dim=1)
        # return F.log_softmax(x, dim=1)
        # out = F.softmax(x, dim=1)
        # out = F.softmax(self.fc_out(feat), dim=1)
        return out

class Simple_Classifier_GTSRB_surrogate(nn.Module):
    def __init__(self):
        super(Simple_Classifier_GTSRB_surrogate, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        # self.fc1 = nn.Linear(256, 256)
        self.fc_feat = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 43)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.relu(F.max_pool2d(self.conv4(x),2))
        # print("Shape before flattening: ", x.shape)
        x = x.view(-1, 256)
        # x = F.relu(self.fc1(x))
        feat = F.relu(self.fc_feat(x))
        # out = F.softmax(self.fc_out(feat), dim=1)
        out = F.log_softmax(self.fc_out(feat), dim=1)
        # return F.log_softmax(x, dim=1)
        # out = F.softmax(self.fc_out(feat), dim=1)
        return out
    
