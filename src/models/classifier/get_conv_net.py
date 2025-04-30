# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://www.kaggle.com/code/sdelecourt/cnn-with-pytorch-for-mnist

import torch
from pathlib import Path
import torch.backends.cudnn as cudnn
from models.classifier.collections.simple_mnist import *
from models.classifier.collections.simple_mnist_magnet import *
from models.classifier.collections.simple_cifar10_v1 import *
from models.classifier.collections.simple_cifar10_v2 import *
from models.classifier.collections.simple_cifar10 import *
from models.classifier.collections.simple_cancer import *
from models.classifier.collections.simple_gtsrb import *
from models.classifier.collections.resnet_cifar10 import *
from models.classifier.collections.simple_fashion import *
from models.classifier.collections.resnet_transfer import *
from models.classifier.collections.simple_speech import *
from models.classifier.collections.badnet import *
from models.classifier.collections.cnn_classifier_3x32x32 import *
from models.classifier.collections.cnn_classifier_1x64x81 import *
from models.classifier.collections.cnn_classifier_1x64x64 import *
from models.classifier.collections.cnn_classifier_3x150x150 import *
from models.classifier.collections.cnn_classifier_1x360x360 import *
from models.classifier.collections.cnn_classifier_1x500x90 import *
from models.classifier.collections.cnn_radioml import *
# from models.classifier import *
from helper import *
def get_conv_net(cfg, model_type = 'target', pre_trained = False):
    
    device = torch.device(cfg.device)
    num_channels=cfg.dataset.num_channels
    num_feats=cfg.dataset.num_feats
    num_classes = cfg.dataset.num_classes



    if cfg.cls_type == 'resnet':
        if cfg.dataset.name in ['mnist', 'fashion', 'gtsrb', 'cancer', 'cifar10', 'speech']:
            net_dict = {
                "target" : get_resnet_model_with_feats(cfg, version='resnet34', pretrained=False, transfer=False),
                "surrogate" :get_resnet_model_with_feats(cfg, version='resnet18', pretrained=False, transfer=False),
                "badnet" : BadNet(num_channels, num_feats, num_classes), #FIXME: Fix the BadNet Model
                }
            net = net_dict[model_type]
            print(f"Starting ResNet classifier for {cfg.dataset.name}_{model_type}")

    elif cfg.cls_type == 'cnn':

        if cfg.dataset.name in ['speech']:
            net_dict = {
                "target" : Classifier1x64x81_target(num_channels, num_feats, num_classes),
                "surrogate" : Classifier1x64x81_surrogate(num_channels, num_feats, num_classes),
                "badnet" : BadNet_Audio(num_channels, num_feats, num_classes),
                }
            net = net_dict[model_type]

        if cfg.dataset.name in ['mnist', 'fashion']:
            net_dict = {
                "target" : Simple_Classifier_MNIST_v1(num_channels, num_feats, num_classes),
                "surrogate" : Simple_Classifier_MNIST_v2(num_channels, num_feats=64, num_classes=10),
                "badnet" : BadNet(num_channels, num_feats, num_classes),
                }
            net = net_dict[model_type]

        if cfg.dataset.name in['cancer']: #cifar10
            net_dict = {
                "target" : Classifier3x150x150_target(num_channels, num_feats, num_classes),
                "surrogate" : Classifier3x150x150_surrogate(num_channels, num_feats, num_classes),
                "badnet" : BadNet(num_channels, num_feats, num_classes), #FIXME: Fix the BadNet Model
                }
            net = net_dict[model_type]


        if cfg.dataset.name in['cifar10', 'gtsrb']:
            net_dict = {
                "target" : Classifier3x32x32_target(num_channels, num_feats, num_classes),
                "surrogate" : Classifier3x32x32_surrogate(num_channels, num_feats, num_classes), 
                "badnet" : BadNet(num_channels, num_feats, num_classes), #FIXME: Fix the BadNet Model
                }
            net = net_dict[model_type]

        if cfg.dataset.name in['radiomlv1', 'radiomlv2', 'radiomlv3']:
            net_dict = {
                "target" : VTCNN2_target(num_channels, num_feats, num_classes),
                "surrogate" : VTCNN2_surrogate(num_channels, num_feats, num_classes), 
                "badnet" : BadNet(num_channels, num_feats, num_classes), #FIXME: Fix the BadNet Model
                }
            net = net_dict[model_type]

        if cfg.dataset.name in['robofi']:
            net_dict = {
                "target" : Classifier_1x360x360_target(num_channels, num_feats, num_classes),
                "surrogate" : Classifier_1x360x360_surrogate(num_channels, num_feats, num_classes), 
                "badnet" : BadNet(num_channels, num_feats, num_classes), #FIXME: Fix the BadNet Model
                }
            net = net_dict[model_type]

    
        if cfg.dataset.name in['activity']:
            net_dict = {
                "target" : Classifier_1x500x90_target(num_channels, num_feats, num_classes),
                "surrogate" : Classifier_1x500x90_surrogate(num_channels, num_feats, num_classes),
                "badnet" : BadNet(num_channels, num_feats, num_classes), #FIXME: Fix the BadNet Model
                }
            net = net_dict[model_type]

    
        if cfg.dataset.name in['medmnist']:
            net_dict = {
                "target" : Classifier1x64x64_target(num_channels, num_feats, num_classes),
                "surrogate" : Classifier1x64x64_surrogate(num_channels, num_feats, num_classes), 
                "badnet" : BadNet(num_channels, num_feats, num_classes), #FIXME: Fix the BadNet Model
                }
            net = net_dict[model_type]
            
    
        
        print(f"Starting CNN classifier for {cfg.dataset.name}_{model_type}")

        
    poison_ratio = cfg.dataset.poison_ratio if model_type=='badnet' else 0
    trigger_label = cfg.dataset.trigger_label if model_type=='badnet' else 0
    num_feats = cfg.dataset.num_feats

    if cfg.retrain:
        print("Overwriting the existing trained model")
        pre_trained = False

    elif pre_trained == True:
        model_dir = Path(f'{cfg.models_dir}/{cfg.dataset.name}_net_{cfg.cls_type}_{model_type}_{num_feats}_{trigger_label}_{poison_ratio}_{cfg.dataset.epochs}.pth')
        if  model_dir.is_file() and not cfg.retrain:
            net = load_state_dict(net, model_dir)
            print('Pre-trained Movel loaded')
        else:
            print("Model does not exists")
            pre_trained = False

    # Use DataParallel if more than one GPU is available
    if cfg.DataParallel and cfg.device == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        
    print(cfg.device)
    net = net.to(device) 

    return net, pre_trained