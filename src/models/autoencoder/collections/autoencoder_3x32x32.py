import torch
import torch.nn as nn

#     This is modified from 'PyTorch-CIFAR-10-autoencoder', from github
#     `PyTorch-CIFAR-10-autoencoder <https://github.com/chenjie/PyTorch-CIFAR-10-autoencoder>`_.
#     """


# Handle "module." prefix if present
def get_autoencoder_3x32x32(cfg):
    class Encoder(nn.Module):
        def __init__(self, input_channels = 3, hidden_layers=3, encoded_space_dim=256):
            super().__init__()

            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
                nn.ReLU(),
                nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
                nn.ReLU(),
                nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
                nn.ReLU(),
                # nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
                # nn.ReLU(),
            )
            

        def forward(self, x):
            x = self.encoder_cnn(x)
            return x

    class Decoder(nn.Module):
        def __init__(self, output_channels = 3, hidden_layers=3, encoded_space_dim=256):
            super().__init__()

            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
                nn.ReLU(),
                nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
                nn.ReLU(),
                nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = self.decoder_conv(x)
            return x 

    encoder = Encoder(cfg.dataset.num_channels, cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    decoder = Decoder(cfg.dataset.num_channels,  cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    return encoder, decoder