import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


def get_autoencoder_simple_mnist_v1(cfg):
    class Encoder(nn.Module):
        def __init__(self, hidden_layers, encoded_space_dim):
            super().__init__()
            ### Convolutional section
            self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #-----
            nn.Conv2d(64, 64, 3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            )
            #----
            ### Flatten layer
            self.flatten = nn.Flatten(start_dim=1)
            ### Linear section
            # self.encoder_lin = nn.Sequential(
            #     # nn.Linear(3 * 3 * 32, 128),
            #     nn.Linear(576, 256),
            #     nn.ReLU(True),
            #     nn.Linear(256, encoded_space_dim)
            # )
            self.encoder_lin = nn.Sequential(
                # nn.Linear(3 * 3 * 32, 128),
                nn.Linear(576, encoded_space_dim),
            )
        def forward(self, x):
            x = self.encoder_cnn(x)
            x = self.flatten(x)
            x = self.encoder_lin(x)
            return x
        
    class Decoder(nn.Module):
        def __init__(self, hidden_layers, encoded_space_dim):
            super().__init__()
            # self.decoder_lin = nn.Sequential(
            #     nn.Linear(encoded_space_dim, 256),
            #     nn.ReLU(True),
            #     # nn.Linear(128, 3 * 3 * 32),
            #     nn.Linear(256, 576),
            #     nn.ReLU(True)
            # )
            self.decoder_lin = nn.Sequential(
                nn.Linear(encoded_space_dim, 576),
                nn.ReLU(True)
            )

            self.unflatten = nn.Unflatten(dim=1,
            unflattened_size=(64, 3, 3))

            self.decoder_conv = nn.Sequential(
                #---
                nn.ConvTranspose2d(64, 64, 3, stride=2, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 3, stride=2, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                #---
                nn.ConvTranspose2d(64, 32, 3,
                stride=2, output_padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 16, 3, stride=2,
                padding=1, output_padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 1, 3, stride=2,
                padding=1, output_padding=1)
            )

        def forward(self, x):
            x = self.decoder_lin(x)
            x = self.unflatten(x)
            x = self.decoder_conv(x)
            # x = torch.sigmoid(x)
            return x 
    encoder = Encoder(cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    decoder = Decoder(cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    return encoder, decoder