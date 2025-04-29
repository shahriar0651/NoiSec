# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_features = 64, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc_feat = nn.Linear(512*block.expansion, num_features)
        # self.out = nn.Linear(512*block.expansion, num_classes)
        self.ft_out = nn.Linear(num_features, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        feat = F.relu(self.fc_feat(x))
        out = self.ft_out(feat)
        return out


def ResNet18(cfg):
    num_feats = cfg.dataset.num_feats
    num_classes = cfg.dataset.num_classes
    return ResNet(BasicBlock, [2, 2, 2, 2], num_features=num_feats, num_classes=num_classes)


def ResNet34(cfg):
    num_feats = cfg.dataset.num_feats
    num_classes = cfg.dataset.num_classes
    return ResNet(BasicBlock, [3, 4, 6, 3], num_features=num_feats, num_classes=num_classes)


def ResNet50(cfg):
    num_feats = cfg.dataset.num_feats
    num_classes = cfg.dataset.num_classes
    return ResNet(Bottleneck, [3, 4, 6, 3], num_features=num_feats, num_classes=num_classes)


def ResNet101(cfg):
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152(cfg):
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test(cfg):
    net = ResNet18(cfg)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# # Load a pre-trained ResNet-50 model
# def pre_Trained_ResNet50(cfg, pretrained = True, retrain = True):
#     model = torchvision.models.resnet50(pretrained=pretrained)
#     # Freeze the pre-trained layers
#     retrain = True
#     num_feats = cfg.dataset.num_feats
#     num_classes = cfg.dataset.num_classes
#     for param in model.parameters():
#         param.requires_grad = retrain # False-->Transfer, True-->Retrain

#     # Replace the classifier (fully connected) layer with a new one
#     in_features = model.fc.in_features
#     model.fc = nn.Sequential( #fc
#         nn.Linear(in_features, num_feats),  # Add a layer to reduce dimensionality to 256
#         nn.ReLU(inplace=True),     # Apply ReLU activation
#         nn.Linear(num_feats, num_classes)         # Final output layer with 10 classes
#     )
#     return model