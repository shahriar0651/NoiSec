import torch
import torch.nn as nn
import torch.nn.functional as F

# # https://github.com/Billy1900/BadNet/blob/main/models/badnet.py
# class BadNet(nn.Module):

#     def __init__(self, input_channels = 1, output_num = 10):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=1),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2, stride=2)
#         )

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2, stride=2)
#         )
#         fc1_input_features = 800 if input_channels == 3 else 512
#         self.fc_feat = nn.Sequential(
#             nn.Linear(in_features=fc1_input_features, out_features=512),
#             nn.ReLU()
#         )
#         self.fc_out = nn.Sequential(
#             nn.Linear(in_features=512, out_features=output_num),
#             nn.Softmax(dim=-1)
#         )
#         self.dropout = nn.Dropout(p=.5)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_feat(x)
#         x = self.fc_out(x)
#         return x
    
# class BadNet(nn.Module):
#     def __init__(self, input_channels = 1, output_num = 10):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=1),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2, stride=2)
#         )

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2, stride=2)
#         )
#         fc1_input_features = 800 if input_channels == 3 else 512
#         self.fc_feat = nn.Sequential(
#             nn.Linear(in_features=fc1_input_features, out_features=512),
#             nn.ReLU()
#         )
#         self.fc_out = nn.Sequential(
#             nn.Linear(in_features=512, out_features=output_num),
#             nn.Softmax(dim=-1)
#         )
#         self.dropout = nn.Dropout(p=.5)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_feat(x)
#         x = self.fc_out(x)
#         return x

class BadNet(nn.Module):
    def __init__(self, num_channels = 1, num_feats=64, num_classes = 10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        fc1_input_features = 800 if num_channels == 3 else 512
        self.fc_feat = nn.Sequential(
            nn.Linear(in_features=fc1_input_features, out_features=num_feats),
            nn.ReLU()
        )
        self.fc_out = nn.Sequential(
            nn.Linear(in_features=num_feats, out_features=num_classes),
            nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc_feat(x)) #TODO: Remove if needed AUChange
        x = self.fc_out(x)
        return x
    

class BadNet_Audio(nn.Module):
    def __init__(self, num_channels = 1, num_feats=64, num_classes = 35):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        fc1_input_features = 800 if num_channels == 3 else 128
        self.fc_feat = nn.Sequential(
            nn.Linear(in_features=fc1_input_features, out_features=num_feats),
            nn.ReLU()
        )
        self.fc_out = nn.Sequential(
            nn.Linear(in_features=num_feats, out_features=num_classes),
            nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc_feat(x)) #TODO: Remove if needed AUChange
        x = self.fc_out(x)
        return x