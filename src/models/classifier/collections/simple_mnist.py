import torch
import torch.nn as nn
import torch.nn.functional as F

# No dropout


class Simple_Classifier_MNIST_v1(nn.Module):
    def __init__(self, num_channels, num_feats, num_classes):
        super(Simple_Classifier_MNIST_v1, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32,64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 512)
        self.fc_feat = nn.Linear(512, num_feats)
        self.fc_out = nn.Linear(num_feats, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = x.view(-1,1600)
        x = F.relu(self.fc1(x))
        feat = F.relu(self.fc_feat(x))
        out = F.softmax(self.fc_out(feat), dim=1)
        return out


class Simple_Classifier_MNIST_v2(nn.Module):
    def __init__(self, num_channels, num_feats, num_classes):
        super(Simple_Classifier_MNIST_v2, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=0)  # input channels = 1, output channels = 64, kernel size = 3x3
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc_feat = nn.Linear(64 * 12 * 12, num_feats)  
        self.fc_out = nn.Linear(num_feats, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        feat = F.relu(self.fc_feat(x))
        x = self.fc_out(feat)
        out = F.softmax(x, dim=1)
        return out
    
# Adding dropout
# class Simple_Classifier_MNIST_v1(nn.Module):
#     def __init__(self):
#         super(Simple_Classifier_MNIST_v1, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=0)  # input channels = 1, output channels = 64, kernel size = 3x3
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.dropout = nn.Dropout(0.5)
#         self.flatten = nn.Flatten()
#         self.fc_feat = nn.Linear(64 * 12 * 12, 128)  
#         self.fc_out = nn.Linear(128, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = self.dropout(x)
#         x = self.flatten(x)
#         feat = F.relu(self.fc_feat(x))
#         x = self.dropout(feat)
#         x = self.fc_out(x)
#         out = F.softmax(x, dim=1)
#         return out
    
# class Simple_Classifier_MNIST_v2(nn.Module):
#     def __init__(self):
#         super(Simple_Classifier_MNIST_v2, self).__init__()
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
#         self.conv3 = nn.Conv2d(32,64, kernel_size=3)
#         self.fc1 = nn.Linear(1600, 128)
#         self.fc_feat = nn.Linear(128, 64)
#         self.fc_out = nn.Linear(64, 10)
#         self.dropout = nn.Dropout(p=0.50)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.relu(F.max_pool2d(self.conv3(x),2))
#         x = x.view(-1,1600)
#         x = self.dropout(F.relu(self.fc1(x)))
#         feat = self.dropout(F.relu(self.fc_feat(x)))
#         out = F.softmax(self.fc_out(feat), dim=1)
#         return out

# class Simple_Classifier_MNIST_v1(nn.Module):
#     def __init__(self):
#         super(Simple_Classifier_MNIST_v1, self).__init__()
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
#         self.conv3 = nn.Conv2d(32,64, kernel_size=5)
#         self.fc1 = nn.Linear(3*3*64, 512)
#         self.fc_feat = nn.Linear(512, 64)
#         self.fc_out = nn.Linear(64, 10)
#         self.dropout = nn.Dropout(p=0.50)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.relu(F.max_pool2d(self.conv3(x),2))
#         x = x.view(-1,3*3*64 )
#         x = self.dropout(F.relu(self.fc1(x)))  # Add 50% dropout
#         feat = self.dropout(F.relu(self.fc_feat(x)))  # Add 50% dropout
#         out = F.softmax(self.fc_out(feat), dim=1)
#         return out

# class Simple_Classifier_MNIST_v2(nn.Module):
#     def __init__(self):
#         super(Simple_Classifier_MNIST_v2, self).__init__()
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
#         self.conv3 = nn.Conv2d(32,64, kernel_size=3)
#         self.fc1 = nn.Linear(1600, 128)
#         self.fc_feat = nn.Linear(128, 64)
#         self.fc_out = nn.Linear(64, 10)
#         self.dropout = nn.Dropout(p=0.50)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.relu(F.max_pool2d(self.conv3(x),2))
#         x = x.view(-1,1600)
#         x = self.dropout(F.relu(self.fc1(x)))
#         feat = self.dropout(F.relu(self.fc_feat(x)))
#         out = F.softmax(self.fc_out(feat), dim=1)
#         return out
    



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Simple_Classifier_MNIST_v1(nn.Module):
#     def __init__(self):
#         super(Simple_Classifier_MNIST_v1, self).__init__()
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
#         self.conv3 = nn.Conv2d(32,64, kernel_size=5)
#         self.fc1 = nn.Linear(3*3*64, 512)
#         self.fc_feat = nn.Linear(512, 64)
#         self.fc_out = nn.Linear(64, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.relu(F.max_pool2d(self.conv3(x),2))
#         x = x.view(-1,3*3*64 )
#         x = F.relu(self.fc1(x))
#         feat = F.relu(self.fc_feat(x))
#         out = F.softmax(self.fc_out(feat), dim=1)
#         # return F.log_softmax(x, dim=1)
#         # out = F.softmax(x, dim=1)
#         return out

# class Simple_Classifier_MNIST_v2(nn.Module):
#     def __init__(self):
#         super(Simple_Classifier_MNIST_v2, self).__init__()
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
#         self.conv3 = nn.Conv2d(32,64, kernel_size=3)
#         self.fc1 = nn.Linear(1600, 128)
#         self.fc_feat = nn.Linear(128, 64)
#         self.fc_out = nn.Linear(64, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.relu(F.max_pool2d(self.conv3(x),2))
#         x = x.view(-1,1600)
#         x = F.relu(self.fc1(x))
#         feat = F.relu(self.fc_feat(x))
#         out = F.softmax(self.fc_out(feat), dim=1)
#         return out
    
