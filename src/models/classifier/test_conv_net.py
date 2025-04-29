import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import json
from helper import *

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test_conv_net(cfg, net, model_type, test_loader):
    device = torch.device(cfg.device)
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    labels = labels.numpy()
    classes = cfg.dataset.classes

    accuracy_dict= {}
    outputs = net(images.to(device))
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu().numpy()


    print('Predicted: ', ' '.join(f'{classes[int(predicted[j])]:5s}'
                                for j in range(4)))
    # Let us look at how the network performs on the whole dataset.
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct // total
    accuracy_dict["Overall"] = accuracy
    print(f'Accuracy of the network on the 10000 test images: {accuracy} %')


    results_dir = file_path(f"{cfg.results_dir}/data/{cfg.dataset.name}_{model_type}_{cfg.dataset.epochs}_accuracy.json")
    with open(results_dir,'w') as fp:
        fp.write(json.dumps(accuracy_dict, indent = 4))

    return accuracy_dict