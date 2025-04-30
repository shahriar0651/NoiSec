# Resources:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://www.kaggle.com/code/sdelecourt/cnn-with-pytorch-for-mnist
# https://blog.paperspace.com/convolutional-autoencoder/

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import torch.backends.cudnn as cudnn
from helper import *
from models.autoencoder.collections import *


def get_autoencoder(cfg, pre_trained = False):
    device = torch.device(cfg.device)
   
    if cfg.dataset.name in ['speech']: 
        if cfg.autoencoder_type == 'CAE':
            encoder, decoder = get_autoencoder_simple_speech_v1(cfg, channels=1)
        elif cfg.autoencoder_type == 'VAE':
            encoder, decoder = get_autoencoder_simple_speech_v1(cfg)

    if cfg.dataset.name in ['mnist', 'fashion']: 
        if cfg.autoencoder_type == 'CAE':
            encoder, decoder = get_autoencoder_simple_mnist_v1(cfg, channels=1)
        elif cfg.autoencoder_type == 'VAE':
            encoder, decoder = get_vae_autoencoder_mnist_fashion(cfg)

    if cfg.dataset.name in ['cifar10', 'gtsrb']:
        encoder, decoder = get_autoencoder_simple_cifar10_v2(cfg, channels=3)

    if cfg.dataset.name in ['cancer']:
        encoder, decoder = get_autoencoder_cancer(cfg) 

    if cfg.dataset.name in ['radiomlv1', 'radiomlv2', 'radiomlv3']:
        encoder, decoder = get_autoencoder_radioml(cfg) 
    if cfg.dataset.name in ['robofi']:
        encoder, decoder = get_autoencoder_robofi(cfg) 
    if cfg.dataset.name in ['medmnist']:
        encoder, decoder = get_autoencoder_1x64x64(cfg) 
    if cfg.dataset.name in ['activity']:
        encoder, decoder = get_autoencoder_1x500x90(cfg) 


    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters in the encoder: {total_params}")
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"Total parameters in the decoder: {total_params}")

    if cfg.retrain:
        print("Overwriting the existing trained model")
        pre_trained = False

    elif pre_trained == True:
        ext = f"{cfg.dataset.ae_hidden_layers}_{cfg.dataset.ae_latent_dim}_{cfg.dataset.noise_factor}_{cfg.dataset.ae_epochs}"
        encoder_dir = Path(f'{cfg.models_dir}/{cfg.dataset.name}_{cfg.autoencoder_type}_encoder_{ext}.pth')
        decoder_dir = Path(f'{cfg.models_dir}/{cfg.dataset.name}_{cfg.autoencoder_type}_decoder_{ext}.pth')
        if encoder_dir.is_file() and decoder_dir.is_file():
            encoder = load_state_dict(encoder, encoder_dir)
            decoder = load_state_dict(decoder, decoder_dir)
            print('Pre-trained Autoencoder loaded')
        else:
            print('Pre-trained Autoencoder is missing')
            print(encoder_dir, "\n", decoder_dir)
            pre_trained = False

    # Use DataParallel if more than one GPU is available
    if cfg.DataParallel and cfg.device == 'cuda' and torch.cuda.device_count() > 1:
        cudnn.benchmark = True
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    return encoder, decoder, pre_trained