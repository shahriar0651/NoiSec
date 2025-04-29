# https://github.com/eugeniaring/Medium-Articles/blob/main/Pytorch/denAE.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from helper import *     

### Training function
def train_epoch_den(cfg, encoder, decoder, device, dataloader, loss_fn, optimizer, noise_factor=0.3, sparse_factor = 0.1):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []

    if cfg.autoencoder_type == 'VAE':
        vae_criterion = nn.BCELoss(reduction='sum')

    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)

        if noise_factor > 0.00:
            image_noisy, _ = add_noise(image_batch, noise_factor)
        else:
            image_noisy = image_batch

        image_batch = image_batch.to(device)
        image_noisy = image_noisy.to(device)

        if cfg.autoencoder_type == 'CAE':
            # Reconstuction Loss 
            encoded_data = encoder(image_noisy)
            decoded_data = decoder(encoded_data)
            loss = loss_fn(decoded_data, image_batch)
        elif cfg.autoencoder_type == 'VAE':
            encoded_data, mu, logvar = encoder(image_noisy)
            decoded_data = decoder(encoded_data)
            bce_loss = vae_criterion(decoded_data, image_batch)
            loss = final_loss(bce_loss, mu, logvar)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss)

### Testing function
def test_epoch_den(cfg, encoder, decoder, device, dataloader, loss_fn, noise_factor=0.3):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_noisy, _ = add_noise(image_batch, noise_factor)
            image_noisy = image_noisy.to(device)
            
            # Encode data
            if cfg.autoencoder_type == 'CAE':
                encoded_data = encoder(image_noisy)
            elif cfg.autoencoder_type == 'VAE':
                encoded_data, _, _ = encoder(image_noisy)
            # Decode data
            decoded_data = decoder(encoded_data)

            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

def plot_ae_outputs_den(cfg, encoder, decoder, train_loader, device, epoch, n=5, noise_factor=0.3):
    plt.figure(figsize=(10,6))
    for i in range(n):
      ax = plt.subplot(4,n,i+1)
      dataiter = iter(train_loader)
      images, _ = next(dataiter)
      img = images[i].unsqueeze(0)
      image_noisy, _ = add_noise(img,noise_factor)     
      image_noisy = image_noisy.to(device)

      encoder.eval()
      decoder.eval()

      with torch.no_grad():
        #   rec_img  = decoder(encoder(image_noisy))
          rec_img = get_reconstructed_image(cfg, encoder, decoder, image_noisy)

      rec_loss = (image_noisy - rec_img)

      data_dict = {'Original images' : scale_to_0_1(img),
                   'Noisy images' : scale_to_0_1(image_noisy),
                   'Reconstructed images' : scale_to_0_1(rec_img),
                   'Reconstruction Loss': scale_to_0_1(rec_loss)}
      
      for indx, (key, val) in enumerate(data_dict.items()):
        #  val  = val.cpu().squeeze().numpy()
         val = np.squeeze(val)
         if val.ndim == 3 and val.shape[0] == 3:
            val = np.transpose(val, (1, 2, 0))
         ax = plt.subplot(4,n, int(i + 1 + indx * n))
         plt.imshow(val, cmap='gist_gray')#, vmin=-1, vmax=1)
         ax.get_xaxis().set_visible(False)
         ax.get_yaxis().set_visible(False)  
         if i == n//2:
            ax.set_title(key)


    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.7, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.3)  
    plt.tight_layout()
    ext = f"{cfg.dataset.ae_hidden_layers}_{cfg.dataset.ae_latent_dim}_{cfg.dataset.noise_factor}_{epoch+1}"
    fig_dir = file_path(f'{cfg.results_dir}/plots/ae_training/ae_training_{ext}.jpg')
    plt.savefig(fig_dir)   
    plt.show()   

    encoder.train()
    decoder.train()

def save_autoencoder_snapshot(cfg, encoder, decoder, ext):
    encoder_dir = file_path(f'{cfg.models_dir}/{cfg.dataset.name}_{cfg.autoencoder_type}_encoder_{ext}.pth')
    decoder_dir = file_path(f'{cfg.models_dir}/{cfg.dataset.name}_{cfg.autoencoder_type}_decoder_{ext}.pth')
    torch.save(encoder.state_dict(), encoder_dir)
    torch.save(decoder.state_dict(), decoder_dir)

def train_autoencoder(cfg, encoder, decoder, train_loader, test_loader):
    device = torch.device(cfg.device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    loss_fn = torch.nn.MSELoss()
    lr = cfg.dataset.lr_ae
    patience = 50

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-09) 

    ### Training cycle
    num_epochs = cfg.dataset.ae_epochs
    noise_factor = cfg.dataset.noise_factor
    sparse_factor = cfg.dataset.sparse_factor 

    best_loss = float('inf')

    history_da={'train_loss':[],'val_loss':[]}

    for epoch in range(num_epochs):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs), end= '\t')
        
        ### Training (use the training function)
        train_loss=train_epoch_den(
            cfg,
            encoder=encoder, 
            decoder=decoder, 
            device=device, 
            dataloader=train_loader, 
            loss_fn=loss_fn, 
            optimizer=optim,
            noise_factor=noise_factor,
            sparse_factor=sparse_factor)
        
        ### Validation  (use the testing function)
        val_loss = test_epoch_den(
            cfg,
            encoder=encoder, 
            decoder=decoder, 
            device=device, 
            dataloader=test_loader, 
            loss_fn=loss_fn,
            noise_factor=noise_factor)
        
        # Print Validationloss
        history_da['train_loss'].append(train_loss)
        history_da['val_loss'].append(val_loss)
        print('train loss {:.3f} \t val loss {:.3f}'.format(train_loss, val_loss))

        plot_ae_outputs_den(cfg, encoder,decoder, test_loader, device, epoch, noise_factor=noise_factor)   
        
        if epoch > 0 and (epoch+1)%25==0:
            ext = f"{cfg.dataset.ae_hidden_layers}_{cfg.dataset.ae_latent_dim}_{cfg.dataset.noise_factor}_{epoch+1}" #{cfg.dataset.sparse_factor}
            save_autoencoder_snapshot(cfg, encoder, decoder, ext)

        # FIXME: Remove early stopping
        # Check for early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping after {epoch+1} epochs without improvement.")
                break

    ext = f"{cfg.dataset.ae_hidden_layers}_{cfg.dataset.ae_latent_dim}_{cfg.dataset.noise_factor}_{cfg.dataset.ae_epochs}"
    save_autoencoder_snapshot(cfg, encoder, decoder, ext)
    return encoder, decoder
