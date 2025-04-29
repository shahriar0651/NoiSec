
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.nn.functional as F


# FIXME: Fixme to load the pre-trained models, and to attach a hook to the feature rep
def get_resnet_model(cfg, version = 'resnet18', pretrained=False, transfer=False):

    num_feats = cfg.dataset.num_feats
    num_classes = cfg.dataset.num_classes

    if version == 'resnet18':
        model = torchvision.models.resnet18(pretrained=pretrained)
    elif version == 'resnet34':
        model = torchvision.models.resnet34(pretrained=pretrained)
    elif version == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)

    # Freeze the pre-trained layers: Transfer-->No grad
    for param in model.parameters():
        param.requires_grad = (not transfer)

    # Replace the classifier (fully connected) layer with a new one
    num_ftrs = model.fc.in_features
    print("Num of features: ", num_ftrs)
    # model.fc = nn.Sequential(
    #     nn.Linear(num_ftrs, num_feats),  # Add a layer to reduce dimensionality to 256
    #     nn.ReLU(inplace=True),          # Apply ReLU activation
    #     nn.Linear(num_feats, num_classes), # Final output layer with 10 classes
    #     nn.Softmax(),          # Adding extra softmax
    # )
    # Update the fully connected layer with dropout
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),  # Dropout with a probability of 50%
        nn.Linear(num_ftrs, num_classes),
        nn.Softmax(dim=1)  # Softmax is applied along the class dimension
    )
    return model

class ResNetWithFeatureExtraction(nn.Module):
    def __init__(self, org_resnet_model, num_feats, num_classes):
        super(ResNetWithFeatureExtraction, self).__init__()
        self.org_resnet_model = org_resnet_model
        
        # Remove the final fully connected layer
        self.features_extractor = nn.Sequential(
            *list(self.org_resnet_model.children())[:-2]  # Keep up to but not including the final fully connected layer
        )
        self.avg_pool = self.org_resnet_model.avgpool
        
        # Add new layers for feature extraction and classification
        self.fc_feat = nn.Linear(self.org_resnet_model.fc.in_features, num_feats)  # Adjust input features to the size after avgpool
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% probability
        self.relu = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(num_feats, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Extract features using the ResNet backbone
        x = self.features_extractor(x)  # Output feature maps
        x = self.avg_pool(x)  # Apply average pooling
        x = torch.flatten(x, 1)  # Flatten the features to (batch_size, num_features)
        # x = x.view(x.size(0), -1)
        x = self.dropout(x)  # Apply dropout

        # Pass through custom fully connected layers with dropout
        features = self.fc_feat(x)  # Intermediate features
        features = self.dropout(features)  # Apply dropout
        features = self.relu(features)  # Apply ReLU activation
        logits = self.fc_out(features)  # Final logits for classification
        outputs = self.softmax(logits)  # Probabilities
        
        # return features, logits, outputs
        return outputs

def get_resnet_model_with_feats(cfg, version='resnet50', pretrained=True, transfer=False):
    num_feats = cfg.dataset.num_feats
    num_classes = cfg.dataset.num_classes

    if version == 'resnet18':
        original_model = models.resnet18(pretrained=pretrained)
    elif version == 'resnet34':
        original_model = models.resnet34(pretrained=pretrained)
    elif version == 'resnet50':
        original_model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported version {version}. Use 'resnet18', 'resnet34', or 'resnet50'.")

    if cfg.dataset.num_channels == 1:
        original_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Freeze the pre-trained layers if transfer learning is not required
    for param in original_model.parameters():
        param.requires_grad = not transfer

    # Create the model with feature extraction capabilities
    model = ResNetWithFeatureExtraction(original_model, num_feats, num_classes)
    return model