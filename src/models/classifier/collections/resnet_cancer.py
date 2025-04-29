import torch.nn as nn
from torchvision import models

def get_resnet_cancer():
    # Load the pre-trained ResNet-18 model
    model = models.resnet18(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer with your own layers
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),  # Add first fully connected layer
        nn.ReLU(),
        nn.Linear(256, 8)  # Add second fully connected layer for 8 classes
    )
    return model