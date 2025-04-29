import torch
import torch.nn as nn


class VTCNN2_target(nn.Module):
    def __init__(self,  num_channels = 3, num_features=256, num_classes = 11):
        super(VTCNN2_target, self).__init__()

        dropout = 0.50

        # Define layers individually for step-by-step application in forward
        self.zero_pad1 = nn.ZeroPad2d(padding=(2, 2, 0, 0))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(1, 3), stride=1, padding=0, bias=True)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)

        self.zero_pad2 = nn.ZeroPad2d(padding=(2, 2, 0, 0))
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=80, kernel_size=(2, 3), stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)

        self.flatten = nn.Flatten()
        self.fc_feat = nn.Linear(in_features=10560, out_features=num_features, bias=True)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout)

        self.fc_out = nn.Linear(in_features=num_features, out_features=num_classes, bias=True)

    def forward(self, x):

        x = self.zero_pad1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.zero_pad2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        feat = self.fc_feat(x)
        feat = self.relu3(feat)
        feat = self.dropout3(feat)
        out = self.fc_out(feat)
        return out
    

class VTCNN2_surrogate(nn.Module):
    def __init__(self,  num_channels = 3, num_features=256, num_classes = 11):
        super(VTCNN2_surrogate, self).__init__()

        dropout = 0.50

        # Define layers individually for step-by-step application in forward
        self.zero_pad1 = nn.ZeroPad2d(padding=(2, 2, 0, 0))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 3), stride=1, padding=0, bias=True)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)

        self.zero_pad2 = nn.ZeroPad2d(padding=(2, 2, 0, 0))
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(2, 3), stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)

        self.flatten = nn.Flatten()
        self.fc_feat = nn.Linear(in_features=8448, out_features=num_features, bias=True)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout)
        self.fc_out = nn.Linear(in_features=num_features, out_features=num_classes, bias=True)

    def forward(self, x):

        x = self.zero_pad1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.zero_pad2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
    
        feat = self.fc_feat(x)
        feat = self.relu3(feat)
        feat = self.dropout3(feat)
        out = self.fc_out(feat)
        return out