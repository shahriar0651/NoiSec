import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import torch.backends.cudnn as cudnn



# Handle "module." prefix if present
def get_autoencoder_simple_mnist_v1(cfg, channels = 1):
    class Encoder(nn.Module):
        def __init__(self, input_channels = 1, hidden_layers=3, encoded_space_dim=256):
            super().__init__()

            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(input_channels, 16, 3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 3, stride=2, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
            
            self.deep_layer = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(64, 64, 3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                ) for _ in range(hidden_layers)
            ])

            self.flatten = nn.Flatten(start_dim=1)
            
            self.encoder_lin = nn.Sequential(
                nn.Linear(576, encoded_space_dim),
            )

        def forward(self, x):
            x = self.encoder_cnn(x)
            for block in self.deep_layer:
                x = block(x)
            x = self.flatten(x)
            x = self.encoder_lin(x)
            return x

    class Decoder(nn.Module):
        def __init__(self, output_channels = 1, hidden_layers=3, encoded_space_dim=256):
            super().__init__()

            self.decoder_lin = nn.Sequential(
                nn.Linear(encoded_space_dim, 576),
                nn.ReLU(True),
            )

            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 3, 3))

            self.deep_layer = nn.ModuleList([
                nn.Sequential(
                    nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                ) for _ in range(hidden_layers)
            ])

            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, output_channels, 3, stride=2, padding=1, output_padding=1),
            )

        def forward(self, x):
            x = self.decoder_lin(x)
            x = self.unflatten(x)
            for block in self.deep_layer:
                x = block(x)
            x = self.decoder_conv(x)
            # x = torch.sigmoid(x)
            return x 

    encoder = Encoder(channels, cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    decoder = Decoder(channels,  cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    return encoder, decoder

def get_autoencoder_simple_speech_v1(cfg, channels = 1):
    class Encoder(nn.Module):
        def __init__(self, input_channels=1, hidden_layers=3, encoded_space_dim=256):
            super().__init__()

            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(input_channels, 16, 3, stride=2, padding=1),  # 1x64x81 -> 16x32x41
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 16x32x41 -> 32x16x21
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 3, stride=2, padding=0),  # 32x16x21 -> 64x7x10
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )

            self.deep_layer = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(64, 64, 3, stride=1, padding=1),  # 64x7x10 -> 64x7x10
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                ) for _ in range(hidden_layers)
            ])

            self.flatten = nn.Flatten(start_dim=1)

            self.encoder_lin = nn.Sequential(
                nn.Linear(64 * 7 * 10, encoded_space_dim),
            )

        def forward(self, x):
            x = self.encoder_cnn(x)
            for block in self.deep_layer:
                x = block(x)
            x = self.flatten(x)
            x = self.encoder_lin(x)
            return x
        
    class Decoder(nn.Module):
        def __init__(self, output_channels=1, hidden_layers=3, encoded_space_dim=256):
            super().__init__()

            self.decoder_lin = nn.Sequential(
                nn.Linear(encoded_space_dim, 64 * 8 * 11),
                nn.ReLU(True),
            )

            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 8, 11))

            self.deep_layer = nn.ModuleList([
                nn.Sequential(
                    nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),  # 64x8x11 -> 64x8x11
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                ) for _ in range(hidden_layers)
            ])

            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=(1, 0)),  # 64x8x11 -> 32x16x22
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=(1, 0)),  # 32x16x22 -> 16x32x44
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, output_channels, 3, stride=2, padding=1, output_padding=(1, 0)),  # 16x32x44 -> 1x64x88
            )

        def forward(self, x):
            x = x.float()  # Ensure the input is of type float
            x = self.decoder_lin(x)
            x = self.unflatten(x)
            for block in self.deep_layer:
                x = block(x)
            x = self.decoder_conv(x)
            x = x[:, :, :64, :81]  # Crop to 1x64x81
            return x 

    encoder = Encoder(channels, cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    decoder = Decoder(channels,  cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    return encoder, decoder

def get_vae_autoencoder_mnist_fashion(cfg):


    class Crop2d(nn.Module):
        def __init__(self, crop_size):
            super(Crop2d, self).__init__()
            self.crop_size = crop_size

        def forward(self, x):
            _, _, h, w = x.size()
            start_h = (h - self.crop_size[0]) // 2
            start_w = (w - self.crop_size[1]) // 2
            return x[:, :, start_h:start_h + self.crop_size[0], start_w:start_w + self.crop_size[1]]

    
    # Define a Conv VAE Encoder
    class ConvVAE_Encoder(nn.Module):
        def __init__(self, encoded_space_dim=256):
            super(ConvVAE_Encoder, self).__init__()
            self.kernel_size = 3  # (4, 4) kernel
            self.init_channels = 8  # initial number of filters
            self.image_channels = 1  # MNIST images are grayscale
            self.latent_dim = encoded_space_dim  # latent dimension for sampling

            # encoder
            self.enc1 = nn.Conv2d(
                in_channels=self.image_channels, out_channels=self.init_channels, kernel_size=self.kernel_size,
                stride=2, padding=1
            )
            self.enc2 = nn.Conv2d(
                in_channels=self.init_channels, out_channels=self.init_channels*2, kernel_size=self.kernel_size,
                stride=2, padding=1
            )
            self.enc3 = nn.Conv2d(
                in_channels=self.init_channels*2, out_channels=self.init_channels*4, kernel_size=self.kernel_size,
                stride=2, padding=1
            )
            self.enc4 = nn.Conv2d(
                in_channels=self.init_channels*4, out_channels=64, kernel_size=self.kernel_size,
                stride=1, padding=1
            )
            # fully connected layers for learning representations
            self.fc1 = nn.Linear(1024, 512)  # Adjust for the 2x2 feature map
            self.fc_mu = nn.Linear(512, self.latent_dim)
            self.fc_log_var = nn.Linear(512, self.latent_dim)

        def reparameterize(self, mu, log_var):
            """
            :param mu: mean from the encoder's latent space
            :param log_var: log variance from the encoder's latent space
            """
            std = torch.exp(0.5 * log_var)  # standard deviation
            eps = torch.randn_like(std)  # `randn_like` as we need the same size
            sample = mu + (eps * std)  # sampling
            return sample

        def forward(self, x):
            # encoding
            x = F.relu(self.enc1(x))
            x = F.relu(self.enc2(x))
            x = F.relu(self.enc3(x))
            x = F.relu(self.enc4(x))
            x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64*2*2)
            hidden = F.relu(self.fc1(x))
            # get `mu` and `log_var`
            mu = self.fc_mu(hidden)
            log_var = self.fc_log_var(hidden)
            # get the latent vector through reparameterization
            z = self.reparameterize(mu, log_var)


            return z, mu, log_var

    # Define a Conv VAE Decoder
    class ConvVAE_Decoder(nn.Module):
        def __init__(self, encoded_space_dim=256):
            super(ConvVAE_Decoder, self).__init__()
            self.kernel_size = 4  # (4, 4) kernel
            self.init_channels = 8  # initial number of filters
            self.image_channels = 1  # MNIST images are grayscale
            self.latent_dim = encoded_space_dim  # latent dimension for sampling

            self.fc2 = nn.Linear(self.latent_dim, 64 * 2 * 2)  # Adjust for the 2x2 feature map
            # decoder
            self.dec1 = nn.ConvTranspose2d(
                in_channels=64, out_channels=self.init_channels*8, kernel_size=self.kernel_size,
                stride=2, padding=1
            )
            self.dec2 = nn.ConvTranspose2d(
                in_channels=self.init_channels*8, out_channels=self.init_channels*4, kernel_size=self.kernel_size,
                stride=2, padding=1
            )
            self.dec3 = nn.ConvTranspose2d(
                in_channels=self.init_channels*4, out_channels=self.init_channels*2, kernel_size=self.kernel_size,
                stride=2, padding=1
            )
            self.dec4 = nn.ConvTranspose2d(
                in_channels=self.init_channels*2, out_channels=self.init_channels*2, kernel_size=self.kernel_size,
                stride=2, padding=1
            )
            self.dec5 = nn.ConvTranspose2d(
                in_channels=self.init_channels*2, out_channels=self.image_channels, kernel_size=self.kernel_size,
                stride=1, padding=1
            )
            self.crop = Crop2d((28, 28))  # Additional layer to crop 32x32 to 28x28
        
        def forward(self, x):
            z = F.relu(self.fc2(x))
            z = z.view(-1, 64, 2, 2)  # Adjust for the 2x2 feature map
            # decoding
            x = F.relu(self.dec1(z))
            x = F.relu(self.dec2(x))
            x = F.relu(self.dec3(x))
            x = F.relu(self.dec4(x))
            x = self.crop(self.dec5(x))
            reconstruction = torch.sigmoid(x)
            return reconstruction
        
    encoder = ConvVAE_Encoder(cfg.dataset.ae_latent_dim) #channels, cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    decoder = ConvVAE_Decoder(cfg.dataset.ae_latent_dim) # channels,  cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    cfg.autoencoder_type = 'VAE'
    return encoder, decoder
    
def get_autoencoder_simple_cifar10_v1(cfg, channels = 1):
    class Encoder(nn.Module):
        def __init__(self, input_channels = 1, hidden_layers=3, encoded_space_dim=256):
            super().__init__()

            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 3, stride=2, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
            
            self.deep_layer = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(64, 64, 3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                ) for _ in range(hidden_layers)
            ])

            self.flatten = nn.Flatten(start_dim=1)
            
            self.encoder_lin = nn.Sequential(
                nn.Linear(1024, encoded_space_dim),
            )

        def forward(self, x):
            x = self.encoder_cnn(x)
            for block in self.deep_layer:
                x = block(x)
            x = self.flatten(x)
            x = self.encoder_lin(x)
            return x

    class Decoder(nn.Module):
        def __init__(self, output_channels = 1, hidden_layers=3, encoded_space_dim=256):
            super().__init__()

            self.decoder_lin = nn.Sequential(
                nn.Linear(encoded_space_dim, 1024),
                nn.ReLU(True),
            )

            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 3, 3))

            self.deep_layer = nn.ModuleList([
                nn.Sequential(
                    nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                ) for _ in range(hidden_layers)
            ])

            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, output_channels, 3, stride=1, padding=1, output_padding=1),
            )

        def forward(self, x):
            x = self.decoder_lin(x)
            x = self.unflatten(x)
            for block in self.deep_layer:
                x = block(x)
            x = self.decoder_conv(x)
            # x = torch.sigmoid(x)
            return x 

    encoder = Encoder(channels, cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    decoder = Decoder(channels,  cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    return encoder, decoder

def get_autoencoder_magnet_cifar10(cfg, channels = 3):
    # Reference:
    # https://github.com/gokulkarthik/MagNet.pytorch/blob/main/defensive_models.py
    class Encoder(nn.Module):
        """
        Defensive model used for CIFAR-10 in MagNet paper
        """
        def __init__(self, input_channels = 3, hidden_layers=3, encoded_space_dim=256):
            super(Encoder, self).__init__()
            self.conv_11 = nn.Conv2d(in_channels=input_channels, out_channels=3, kernel_size=3, stride=1, padding=1)
            self.conv_21 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        def forward(self, X):
            """Forward propagation
            :param X: Mini-batch of shape [-1, 1, H, W]
            :return: Mini-batch of shape [-1, 1, H, W]
            """
            X = torch.sigmoid(self.conv_11(X))
            X = torch.sigmoid(self.conv_21(X))
            return X
    class Decoder(nn.Module):
        """
        Defensive model used for CIFAR-10 in MagNet paper
        """
        def __init__(self, input_channels = 3, hidden_layers=3, encoded_space_dim=256):
            super(Decoder, self).__init__()
            self.conv_31 = nn.Conv2d(in_channels=3, out_channels=input_channels, kernel_size=3, stride=1, padding=1)

        def forward(self, X):
            """Forward propagation

            :param X: Mini-batch of shape [-1, 1, H, W]
            :return: Mini-batch of shape [-1, 1, H, W]
            """
            X = torch.sigmoid(self.conv_31(X))
            return X
    
    encoder = Encoder(channels, cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    decoder = Decoder(channels,  cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    return encoder, decoder

def get_autoencoder_simple_cifar10_v2(cfg, channels = 1):
    class Encoder(nn.Module):
        def __init__(self, in_channels=3, out_channels=16, encoded_space_dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1), # (32, 32)
                nn.ReLU(True),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1), 
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, 2*out_channels, 3, padding=1, stride=2), # (16, 16)
                nn.BatchNorm2d(2*out_channels),
                nn.ReLU(True),
                nn.Conv2d(2*out_channels, 2*out_channels, 3, padding=1),
                nn.BatchNorm2d(2*out_channels),
                nn.ReLU(True),
                nn.Conv2d(2*out_channels, 4*out_channels, 3, padding=1, stride=2), # (8, 8)
                nn.BatchNorm2d(4*out_channels),
                nn.ReLU(True),
                nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=1),
                nn.BatchNorm2d(4*out_channels),
                nn.ReLU(True),
                nn.Flatten(),
                nn.Linear(4*out_channels*8*8, encoded_space_dim),
                nn.ReLU(True)
            )

        def forward(self, x):
            x = x.view(-1, 3, 32, 32)
            output = self.net(x)
            return output

    #  defining decoder
    class Decoder(nn.Module):
        def __init__(self, in_channels=3, out_channels=16, encoded_space_dim=256):
            super().__init__()
            self.out_channels = out_channels

            self.linear = nn.Sequential(
                nn.Linear(encoded_space_dim, 4*out_channels*8*8),
                nn.ReLU(True)
            )

            self.conv = nn.Sequential(
                nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1), # (8, 8)
                nn.BatchNorm2d(4*out_channels),
                nn.ReLU(True),
                nn.ConvTranspose2d(4*out_channels, 2*out_channels, 3, padding=1, 
                                stride=2, output_padding=1), # (16, 16)
                nn.BatchNorm2d(2*out_channels),
                nn.ReLU(True),
                nn.ConvTranspose2d(2*out_channels, 2*out_channels, 3, padding=1),
                nn.BatchNorm2d(2*out_channels),
                nn.ReLU(True),
                nn.ConvTranspose2d(2*out_channels, out_channels, 3, padding=1, 
                                stride=2, output_padding=1), # (32, 32)
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.ConvTranspose2d(out_channels, in_channels, 3, padding=1)
            )

        def forward(self, x):
            output = self.linear(x)
            output = output.view(-1, 4*self.out_channels, 8, 8)
            output = self.conv(output)
            # output = torch.sigmoid(output)
            return output

    encoder = Encoder(in_channels=cfg.dataset.num_channels, 
                      out_channels=int((cfg.dataset.ae_hidden_layers-2)*8), # 3-->8, 4-->16, 5-->32
                      encoded_space_dim=cfg.dataset.ae_latent_dim)
    decoder = Decoder(in_channels=cfg.dataset.num_channels, 
                      out_channels=int((cfg.dataset.ae_hidden_layers-2)*8), # 3-->8, 4-->16, 5-->32
                      encoded_space_dim=cfg.dataset.ae_latent_dim)
    return encoder, decoder

def get_autoencoder_simple_syncan_v1(cfg, channels = 1):
    class Encoder(nn.Module):
        def __init__(self, input_channels = 1, hidden_layers=3, encoded_space_dim=256):
            super().__init__()

            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(input_channels, 16, 3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 3, stride=2, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
            
            self.deep_layer = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(64, 64, 3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                ) for _ in range(hidden_layers)
            ])

            self.flatten = nn.Flatten(start_dim=1)
            
            self.encoder_lin = nn.Sequential(
                nn.Linear(256, encoded_space_dim),
            )

        def forward(self, x):
            x = self.encoder_cnn(x)
            for block in self.deep_layer:
                x = block(x)
            x = self.flatten(x)
            x = self.encoder_lin(x)
            return x

    class Decoder(nn.Module):
        def __init__(self, output_channels = 1, hidden_layers=3, encoded_space_dim=256):
            super().__init__()

            self.decoder_lin = nn.Sequential(
                nn.Linear(encoded_space_dim, 256),
                nn.ReLU(True),
            )

            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 2, 2))

            self.deep_layer = nn.ModuleList([
                nn.Sequential(
                    nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                ) for _ in range(hidden_layers)
            ])

            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, output_channels, 3, stride=2, padding=1, output_padding=1),
            )

        def forward(self, x):
            x = self.decoder_lin(x)
            x = self.unflatten(x)
            for block in self.deep_layer:
                x = block(x)
            x = self.decoder_conv(x)
            return x 

    encoder = Encoder(channels, cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    decoder = Decoder(channels,  cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    return encoder, decoder

def get_autoencoder_cancer(cfg):
    class Encoder(nn.Module):
        def __init__(self, encoded_space_dim=256):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1),   # 150x150x3 -> 75x75x16
                nn.ReLU(True),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 75x75x16 -> 38x38x32
                nn.ReLU(True),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 38x38x32 -> 19x19x64
                nn.ReLU(True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 19x19x64 -> 10x10x128
                nn.ReLU(True),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 10x10x128 -> 5x5x256
                nn.ReLU(True),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, encoded_space_dim, 3, stride=2, padding=1),  # 5x5x256 -> 3x3xencoded_space_dim
                nn.ReLU(True)
            )

        def forward(self, x):
            output = self.encoder(x)
            return output

    class Decoder(nn.Module):
        def __init__(self,  encoded_space_dim=256):
            super().__init__()
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(encoded_space_dim, 256, 4, stride=2, padding=1),  # 3x3xencoded_space_dim -> 6x6x256
                nn.ReLU(True),
                nn.BatchNorm2d(256),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, output_padding=1),  # 6x6x256 -> 12x12x128
                nn.ReLU(True),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=1),  # 12x12x128 -> 24x24x64
                nn.ReLU(True),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=1),  # 24x24x64 -> 48x48x32
                nn.ReLU(True),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, output_padding=1),  # 48x48x32 -> 96x96x16
                nn.ReLU(True),
                nn.BatchNorm2d(16),
                nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1, output_padding=1),   # 96x96x16 -> 192x192x3
                nn.Sigmoid()  # Ensures output values are in the range [0, 1]
                )

        def forward(self, x):
            output = self.decoder(x)
            decoded = output[:, :, :150, :150]
            return decoded

    # Create instances of Encoder and Decoder
    encoder = Encoder(encoded_space_dim=cfg.dataset.ae_latent_dim)
    decoder = Decoder(encoded_space_dim=cfg.dataset.ae_latent_dim)
    return encoder, decoder


def get_autoencoder_radioml(cfg):
    class Encoder(nn.Module):
        def __init__(self, encoded_space_dim=256):
            super().__init__()
            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),  # 1x2x128 -> 16x2x64
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), # 16x2x64 -> 32x2x32
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), # 32x2x32 -> 64x2x16
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), # 64x2x16 -> 128x2x8
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))  # 128x2x8 -> 256x2x4
                )
            self.flatten = nn.Flatten()
            self.encoder_lin = nn.Linear(256 * 2 * 4, encoded_space_dim)

        def forward(self, x):
            x = self.encoder_cnn(x)
            x = self.flatten(x)
            encoded = self.encoder_lin(x)
            return encoded

    class Decoder(nn.Module):
        def __init__(self,  encoded_space_dim=256):
            super().__init__()
            self.decoder_linear = nn.Linear(encoded_space_dim, 256 * 2 * 4)
            self.decoder_cnn = nn.Sequential(
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)), # 256x2x4 -> 128x2x8
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),  # 128x2x8 -> 64x2x16
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),   # 64x2x16 -> 32x2x32
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),  # 32x2x32 -> 16x2x64
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),   # 16x2x64 -> 1x2x128
                nn.Sigmoid()  # Use sigmoid to scale the output between 0 and 1
            )

        def forward(self, x):
            x = self.decoder_linear(x)
            x = x.view(-1, 256, 2, 4)
            decoded = self.decoder_cnn(x)
            return decoded

    # Create instances of Encoder and Decoder
    encoder = Encoder(encoded_space_dim=cfg.dataset.ae_latent_dim)
    decoder = Decoder(encoded_space_dim=cfg.dataset.ae_latent_dim)
    return encoder, decoder


def get_autoencoder_robofi(cfg, channels = 1):
    class Encoder(nn.Module):
        def __init__(self, input_channels=1, hidden_layers=1, encoded_space_dim=256):
            super().__init__()

            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(input_channels, 16, 3, stride=2, padding=1),  # 1x64x81 -> 16x32x41
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 16x32x41 -> 32x16x21
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 3, stride=2, padding=0),  # 32x16x21 -> 64x7x10
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 128, 3, stride=2, padding=0),  # 32x16x21 -> 64x7x10
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(128, 128, 3, stride=2, padding=0),  # 32x16x21 -> 64x7x10
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(128, 256, 3, stride=2, padding=0),  # 32x16x21 -> 64x7x10
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )

            self.deep_layer = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(64, 64, 3, stride=1, padding=1),  # 64x7x10 -> 64x7x10
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                ) for _ in range(hidden_layers)
            ])

            self.flatten = nn.Flatten(start_dim=1)

            self.encoder_lin = nn.Sequential(
                nn.Linear(4096, encoded_space_dim),
            )

        def forward(self, x):
            x = self.encoder_cnn(x)
            # for block in self.deep_layer:
            #     x = block(x)
            #     print(x.shape)
            x = self.flatten(x)
            x = self.encoder_lin(x)
            return x
    class Decoder(nn.Module):
        def __init__(self, output_channels=1, hidden_layers=3, encoded_space_dim=256):
            super().__init__()

            self.decoder_lin = nn.Sequential(
                nn.Linear(encoded_space_dim, 4096),
                nn.ReLU(True),
            )

            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 4, 4))

            # self.deep_layer = nn.ModuleList([
            #     nn.Sequential(
            #         nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),  # 64x8x11 -> 64x8x11
            #         nn.BatchNorm2d(64),
            #         nn.ReLU(True),
            #     ) for _ in range(hidden_layers)
            # ])

            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1,), # output_padding=(1, 1)),  # 64x8x11 -> 32x16x22
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1), # output_padding=(1, 1)),  # 64x8x11 -> 32x16x22
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1), # output_padding=(1, 1)),  # 64x8x11 -> 32x16x22
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1), # output_padding=(1, 1)),  # 32x16x22 -> 16x32x44
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1), # output_padding=(1, 1)),  # 32x16x22 -> 16x32x44
                nn.BatchNorm2d(32),
                nn.ReLU(True),            
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1), # output_padding=(1, 1)),  # 32x16x22 -> 16x32x44
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, output_channels, 3, stride=2, padding=1), # output_padding=(1, 1)),  # 16x32x44 -> 1x64x88
            )

        def forward(self, x):
            x = x.float()  # Ensure the input is of type float
            x = self.decoder_lin(x)
            x = self.unflatten(x)
            # for block in self.deep_layer:
            #     x = block(x)
            x = self.decoder_conv(x)
            x = x[:, :, :360, :360]  # Crop to 1x64x81
            return x 

    encoder = Encoder(channels, cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    decoder = Decoder(channels,  cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    return encoder, decoder


def get_autoencoder_1x64x64(cfg, channels = 1):
    class Encoder(nn.Module):
        def __init__(self, input_channels=1, hidden_layers=1, encoded_space_dim=256):
            super().__init__()

            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(input_channels, 16, kernel_size=(3,3), stride=2, padding=1),  # 1x64x81 -> 16x32x41
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.Conv2d(16, 32, kernel_size=(3,3), stride=2, padding=1),  # 16x32x41 -> 32x16x21
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, kernel_size=(3,3), stride=2, padding=0),  # 32x16x21 -> 64x7x10
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 128, kernel_size=(3,3), stride=2, padding=0),  # 32x16x21 -> 64x7x10
                nn.BatchNorm2d(128),
                nn.ReLU(True),
            )

            self.deep_layer = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(64, 64, 3, stride=1, padding=1),  # 64x7x10 -> 64x7x10
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                ) for _ in range(hidden_layers)
            ])

            self.flatten = nn.Flatten(start_dim=1)

            self.encoder_lin = nn.Sequential(
                nn.Linear(1152, encoded_space_dim),
            )

        def forward(self, x):
            x = self.encoder_cnn(x)
            x = self.flatten(x)
            x = self.encoder_lin(x)
            return x
        
    class Decoder(nn.Module):
        def __init__(self, output_channels=1, hidden_layers=3, encoded_space_dim=256):
            super().__init__()

            self.decoder_lin = nn.Sequential(
                nn.Linear(encoded_space_dim, 1152),
                nn.ReLU(True),
            )

            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 3, 3))

            # Use ConvTranspose2d layers to progressively upsample
            self.decoder_conv = nn.Sequential(
                # nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 0)),  # 256x6x2 -> 128x12x4
                # nn.BatchNorm2d(128),
                # nn.ReLU(True),
                
                nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),#  output_padding=(1, 1)),  # 128x12x4 -> 64x24x8
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), #output_padding=(1, 1)),  # 64x24x8 -> 32x48x16
                nn.BatchNorm2d(32),
                nn.ReLU(True),

                nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), #output_padding=(1, 1)),  # 32x48x16 -> 16x96x32
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), #output_padding=(1, 1)),  # 32x48x16 -> 16x96x32
                nn.BatchNorm2d(16),
                nn.ReLU(True),

                nn.ConvTranspose2d(16, output_channels, kernel_size=(3, 3), stride=(2, 2)), # padding=(1, 1), output_padding=(0, 0)),  # 16x96x32 -> 1x192x64
            )

        def forward(self, x):
            x = x.float()  # Ensure the input is of type float
            x = self.decoder_lin(x)
            x = self.unflatten(x)
            x = self.decoder_conv(x)
            x = x[:, :, :64, :64]  # Crop to 1x64x81
            return x 
    encoder = Encoder(channels, cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    decoder = Decoder(channels,  cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    return encoder, decoder


def get_autoencoder_1x500x90(cfg, channels = 1):
    class Encoder(nn.Module):
        def __init__(self, input_channels=1, hidden_layers=1, encoded_space_dim=256):
            super().__init__()

            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(input_channels, 16, kernel_size=(3,3), stride=2, padding=1),  # 1x64x81 -> 16x32x41
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.Conv2d(16, 32, kernel_size=(3,1), stride=2, padding=1),  # 16x32x41 -> 32x16x21
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, kernel_size=(3,1), stride=2, padding=0),  # 32x16x21 -> 64x7x10
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 128, kernel_size=(3,1), stride=2, padding=0),  # 32x16x21 -> 64x7x10
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(128, 128, kernel_size=(3,1), stride=2, padding=0),  # 32x16x21 -> 64x7x10
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(128, 256, kernel_size=(3,1), stride=2, padding=0),  # 32x16x21 -> 64x7x10
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )

            self.deep_layer = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(64, 64, 3, stride=1, padding=1),  # 64x7x10 -> 64x7x10
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                ) for _ in range(hidden_layers)
            ])

            self.flatten = nn.Flatten(start_dim=1)

            self.encoder_lin = nn.Sequential(
                nn.Linear(3072, encoded_space_dim),
            )

        def forward(self, x):
            x = self.encoder_cnn(x)
            x = self.flatten(x)
            x = self.encoder_lin(x)
            return x

    class Decoder(nn.Module):
        def __init__(self, output_channels=1, hidden_layers=3, encoded_space_dim=256):
            super().__init__()

            self.decoder_lin = nn.Sequential(
                nn.Linear(encoded_space_dim, 3072),
                nn.ReLU(True),
            )

            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 6, 2))

            # Use ConvTranspose2d layers to progressively upsample
            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 0)),  # 256x6x2 -> 128x12x4
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 0)),  # 128x12x4 -> 64x24x8
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 0)),  # 64x24x8 -> 32x48x16
                nn.BatchNorm2d(32),
                nn.ReLU(True),

                nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 0)),  # 32x48x16 -> 16x96x32
                nn.BatchNorm2d(16),
                nn.ReLU(True),

                nn.ConvTranspose2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 0)),  # 32x48x16 -> 16x96x32
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 0)),  # 32x48x16 -> 16x96x32
                nn.BatchNorm2d(16),
                nn.ReLU(True),

                nn.ConvTranspose2d(16, output_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0, 0)),  # 16x96x32 -> 1x192x64
            )

        def forward(self, x):
            x = x.float()  # Ensure the input is of type float
            x = self.decoder_lin(x)
            x = self.unflatten(x)
            x = self.decoder_conv(x)
            x = x[:, :, :500, :90]  # Crop to 1x64x81
            return x 
        

    encoder = Encoder(channels, cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    decoder = Decoder(channels,  cfg.dataset.ae_hidden_layers, cfg.dataset.ae_latent_dim)
    return encoder, decoder