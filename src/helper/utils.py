import torch
import numpy as np
from  pathlib import Path
import matplotlib.pyplot as plt
plt.set_loglevel('WARNING')
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.image")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import os
import sys
import time
import math
from torch.utils.data import Dataset
import copy
from matplotlib.colors import Normalize
from torchvision.datasets import ImageFolder

import seaborn as sns
from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

import pandas as pd
from sklearn.manifold import TSNE


def file_path(directory):
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    return directory

def update_abs_path(cfg, source_dir):
    """
    Update absolute paths in the configuration object.

    Parameters:
    - cfg (object): Configuration object with attributes workspace_dir, datasets_dir, models_dir, results_dir, and device.
    - source_dir (str): Source directory used for updating paths.

    Returns:
    object: Updated configuration object.

    This function updates paths in the provided configuration object (`cfg`) based on the given source directory (`source_dir`).
    It modifies the following attributes of `cfg`:
    - `workspace_dir`: Set to the parent directory of the source directory.
    - `datasets_dir`: Set to the concatenation of the source directory and `cfg.dataset_dir`.
    - `models_dir`: Set to the concatenation of the source directory and `cfg.models_dir`.
    - `results_dir`: Set to the concatenation of the source directory and `cfg.results_dir`.

    Directories are created if they do not exist:
    - `cfg.dataset_dir`
    - `cfg.models_dir`
    - `cfg.results_dir`

    The function also sets the `cfg.device` based on CUDA availability and prints the selected device.

    Example:
    ```
    config = update_abs_path(config, '/path/to/source')
    ```

    Note: Assumes that `cfg` has attributes `workspace_dir`, `datasets_dir`, `models_dir`, `results_dir`, and `device`.
    """
    source_dir = Path(source_dir)
    cfg.workspace_dir = source_dir.parent
    cfg.dataset_dir = source_dir / cfg.dataset_dir
    cfg.scaler_dir = source_dir / cfg.scaler_dir
    
     
    cfg.models_dir = source_dir / cfg.models_dir / cfg.dataset.name 

    # List of parameters
    params = [
        cfg.rep, 
        cfg.scale, 
        cfg.dataset.num_feats, 
        cfg.autoencoder_type, 
        cfg.cls_type, 
        cfg.dataset.ae_latent_dim, 
        cfg.dataset.ae_hidden_layers, 
        cfg.dataset.noise_factor,
        # cfg.sync,
        cfg.magsec, 
    ]

    attack_for_adaptive_atk = cfg.adaptive_attack
    if cfg.adaptive:
        cfg.adv_attacks=["RN", cfg.adaptive_attack]
        cfg.dataset.atk_cfg[attack_for_adaptive_atk]['eps'] = cfg.adaptive_eps
        eps_for_adaptive_atk = cfg.dataset.atk_cfg[attack_for_adaptive_atk]['eps']
        params.append(eps_for_adaptive_atk)
    
    print("cfg.adv_attacks: ", cfg.adv_attacks)



    # Convert the parameters into a string
    params_str = "_".join(str(param) for param in params)
    # Construct the results_dir path
    cfg.results_dir = source_dir / cfg.results_dir / cfg.dataset.name / params_str


    cfg.dataset_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    if cfg.DataParallel and cfg.device == 'cuda' and torch.cuda.device_count() > 1:
        cfg.DataParallel=True
    else:
        cfg.DataParallel=False

    print(f"Device: {cfg.device}")
    return cfg

def scale_to_0_1(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    max_value = np.max(tensor)
    min_value = np.min(tensor)
    if max_value > min_value:
        scaled_tensor = (tensor - min_value) / (max_value - min_value)
    else:
        scaled_tensor = tensor
    return torch.tensor(scaled_tensor, dtype=torch.float32) #.to(tensor.device)
    #return scaled_tensor

def clip_to_0_1(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()    
    # Clip from 0 to 1
    clipped_tensor = np.clip(tensor, 0, 1)
    return clipped_tensor

def add_noise(inputs, noise_factor=0.2):
    noises = torch.randn_like(inputs) * noise_factor
    noisy = inputs + noises
    noisy = torch.clip(noisy, 0.0, 1.)
    return noisy, noises



def add_noise_numpy(images, noise_factor):
    # Add random noise to the images
    noisy_images = images + noise_factor * np.random.randn(*images.shape)
    # Clip the values to be between 0 and 1 since images should be in this range
    noisy_images = np.clip(noisy_images, 0., 1.)
    return noisy_images




def plot_images_with_confidences_v2(cfg, accessType, attack_type, combined_conf_pre_feat, env, n=11):

    col_row_pos = {
        'Natural_Org_Noise': (1, 1),
        'Adversarial_Org_Noise': (1, 2),
        'Benign_Org_Noise': (1, 3),
        'Natural_Org_Image': (2, 1),
        'Adversarial_Org_Image': (2, 2),
        'Benign_Org_Image': (2, 3),
        'Natural_Rec_Image': (3, 1),
        'Adversarial_Rec_Image': (3, 2),
        'Benign_Rec_Image': (3, 3),
        'Natural_Rec_Noise': (4, 1),
        'Adversarial_Rec_Noise': (4, 2),
        'Benign_Rec_Noise': (4, 3),
    }

    rename_dict = {
        'Natural_Org_Noise': 'Added\nNoise',
        'Adversarial_Org_Noise': 'Added\nNoise',
        'Benign_Org_Noise': 'Added\nNoise',
        'Natural_Org_Image': 'Test\nInput',
        'Adversarial_Org_Image': 'Test\nInput',
        'Benign_Org_Image': 'Test\nInput',
        'Natural_Rec_Image': 'Recon.\nInput',
        'Adversarial_Rec_Image': 'Recon.\nInput',
        'Benign_Rec_Image': 'Recon.\nInput',
        'Natural_Rec_Noise': 'Recon.\nNoise',
        'Adversarial_Rec_Noise': 'Recon.\nNoise',
        'Benign_Rec_Noise': 'Recon.\nNoise',
    }



    for i, conf_pre_feat in enumerate(combined_conf_pre_feat[0:n]):

        org_img_pred = conf_pre_feat['Original_Org_Image']['Pred']
        nat_img_pred = conf_pre_feat['Natural_Org_Image']['Pred']
        adv_img_pred = conf_pre_feat['Adversarial_Org_Image']['Pred']
        cln_aim_pred = conf_pre_feat['Adversarial_Rec_Image']['Pred']


        att_at_nat = int(org_img_pred != nat_img_pred)
        att_at_adv = int(org_img_pred != adv_img_pred)
        att_at_rec = int(org_img_pred != cln_aim_pred)

        col_adv, col_rec = 'red' if att_at_adv == 1 else 'green', 'red' if att_at_rec == 1 else 'green'

        max_y_scale_img = 0  # Variable to store the maximum y-scale across all rows
        max_y_scale_noi = 0  # Variable to store the maximum y-scale across all rows

        bar_img_index = [5,6,13,14,21,22]
        bar_noi_index = [4,7,12,15,20,23]

        plt.figure(figsize=(10, 10))
        for image_type, image_data in conf_pre_feat.items():

            if image_type not in col_row_pos:
                continue
            col = col_row_pos[image_type][0] - 1
            row = col_row_pos[image_type][1] - 1
            position = row * 4 * 2 + col + 1

            ax = plt.subplot(6, 4, int(position))
            image = np.squeeze(image_data['Image'])
            if image.ndim == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            if 'Image' not in image_type: # Scaling the noise from 0-1
                image = scale_to_0_1(image)
            else:
                image = clip_to_0_1(image)
            plt.imshow(image, aspect='equal')  #, aspect='auto') #, cbar=False, vmin=vmin, vmax=vmax)
            # Ensure the axes are set to square dimensions
            plt.gca().set_aspect('equal', adjustable='box')

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # ax.set_title(image_type.replace("_", "\n"), fontsize='12')
            title_str = f"{rename_dict[image_type]} [{image_data['Pred']}]"
            ax.set_title(title_str, fontsize='12')

            ax = plt.subplot(6, 4, int(position + 4))
            data = image_data[cfg.rep]
            sns.barplot(data)  # color=data[1]) #TODO: Add color bar to correct or incorrect prediction
            # Plot the barplot
            # sns.barplot(data, ax=ax)

            # # Force the subplot to have a square aspect ratio
            # ax.set_aspect('equal', adjustable='box')

            if data.shape[0] <= 10:
                ax.get_xaxis().set_visible(True)
            else:
                ax.get_xaxis().set_visible(False)

            ax.get_yaxis().set_visible(True)
            
            if 'Image' in image_type:
                max_y_scale_img = max(max_y_scale_img, max(data))  # Update the maximum y-scale
            else:
                max_y_scale_noi = max(max_y_scale_noi, max(data))  # Update the maximum y-scale

        # Set the same y-scale for all rows
        for img_indx, noi_indx in zip(bar_img_index, bar_noi_index):
            plt.subplot(6, 4, img_indx+1).set_ylim([0, max_y_scale_img])
            plt.subplot(6, 4, noi_indx+1).set_ylim([0, max_y_scale_noi])

        plt.tight_layout()
        print(cfg.results_dir, accessType, attack_type, env, i + 1, att_at_nat, att_at_adv, att_at_rec)
        fig_dir = Path(
            '{}/plots/{}_{}_detection/comb_img_cof_{}_{}_{}{}{}.jpg'.format(
                cfg.results_dir, accessType, attack_type, env, i + 1, att_at_nat, att_at_adv, att_at_rec))
        fig_dir.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(fig_dir, dpi=cfg.fig_dpi)
        plt.show()

def plot_images_with_confidences(cfg, attack_type, pred_dict, data_dict, conf_dict, epsilon, env, n=11):
    """
    Plot images with confidences for a given attack scenario.

    Parameters:
    - cfg (object): Configuration object.
    - attack_type (str): Type of attack.
    - pred_dict (dict): Dictionary containing prediction information.
    - data_dict (dict): Dictionary containing image and noise data.
    - conf_dict (dict): Dictionary containing confidence values.
    - epsilon (float): Attack epsilon value.
    - env (str): Environment information.
    - n (int): Number of images to plot. Default is 11.

    Returns:
    None

    This function plots images and associated confidences for a given attack scenario.
    It generates plots for original images, adversarial images, and reconstructed images,
    along with confidence values for each component.

    The plots include:
    - Original image
    - Adversarial noise
    - Adversarial image
    - Reconstructed image
    - Reconstructed noise

    Confidence values are represented bar plots.

    Example:
    ```python
    plot_images_with_confidences(cfg, 'attack_type', pred_dict, data_dict, conf_dict, 0.1, 'dev', n=5)
    ```
    """    

    # Make the values 2D (W x H) or 3D ( W x H x C)
    for key, values in data_dict.items():
        new_values = []
        for val in values:
            val = np.squeeze(val)
            if val.ndim == 3 and val.shape[0] == 3:
                val = np.transpose(val, (1, 2, 0))
            new_values.append(val)
        data_dict[key] = new_values

    for i in range(n):
        plt.figure(figsize=(10, 2.5))


        ben_img_pred, adv_img_pred, cln_aim_pred = (
            pred_dict['ben_img_preds'][i],
            pred_dict['adv_img_preds'][i],
            pred_dict['cln_aim_preds'][i],
        )

        plot_images = {
            'Benign\nImage': data_dict['ben_images'][i],
            'Reconstructed\nBen Image': data_dict['cln_bnimgs'][i],
            'Reconstructed\nBen Noise': data_dict['ben_noises'][i],
            'Adversarial\nPerturbation': data_dict['adv_prtrbs'][i],
            'Adversarial\nImage': data_dict['adv_images'][i],
            'Reconstructed\nAdv Image': data_dict['cln_adimgs'][i],
            'Reconstructed\nAdv Noise': data_dict['adv_noises'][i],
        }

        att_at_adv, att_at_rec = int(ben_img_pred != adv_img_pred), int(ben_img_pred != cln_aim_pred)
        col_adv, col_rec = 'red' if att_at_adv == 1 else 'green', 'red' if att_at_rec == 1 else 'green'

        conf_keys_colors = {
            'ben_img_conf': 'blue',
            'cln_bim_conf': 'blue',
            'ben_noi_conf': 'gray',
            'adv_prt_conf': 'gray',
            'adv_img_conf': col_adv,
            'cln_aim_conf': col_rec,
            'adv_noi_conf': 'orange',
        }

        plot_confs = {key: [conf_dict[key][i], color] for key, color in conf_keys_colors.items()}


        plot_data_list = [plot_images, plot_confs]

        for row, plot_data in enumerate(plot_data_list):
            for col, (title, data) in enumerate(plot_data.items()):
                ax = plt.subplot(len(plot_data_list), len(plot_data_list[0]), int(row * len(plot_data_list[0]) + col + 1))
                if row == 0:
                    vmin, vmax = np.floor(np.min(data)), np.ceil(np.max(data))
                    # sns.heatmap(data, square=True, cbar=False, vmin=vmin, vmax=vmax)
                    plt.imshow(data) #, square=True, cbar=False, vmin=vmin, vmax=vmax)
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.set_title(title, fontsize = '10')
                elif row == 1:
                    sns.barplot(data[0], color=data[1])
                    # plt.yticks([0, 0.25, 0.5, 0.75, 1])
                    # plt.xticks(fontsize = '8')
                    if data[0].shape[0] <= 10:
                        ax.get_xaxis().set_visible(True)
                    else:
                        ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

        plt.tight_layout()
        fig_dir = Path('{}/plots/{}_detection/comb_img_cof_{}_{:.3f}_{}_{}{}.jpg'.format(
            cfg.results_dir, attack_type, env, epsilon, i + 1, att_at_adv, att_at_rec), dpi=350)
        fig_dir.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(fig_dir)
        plt.show()


def plot_ae_outputs_den(cfg, attack_type, data_dict, conf_dict, epsilon, env, n=11):
    # Plot the confidences
    plt.figure(figsize=(8,6))
    for i in range(n):
        ben_image = data_dict['ben_images'][i]
        adv_image = data_dict['adv_images'][i] 
        cln_image = data_dict['cln_images'][i] 
        org_noise = data_dict['org_noises'][i]
        adv_noise = data_dict['adv_prtrbs'][i]
        adv_noise = data_dict['adv_noises'][i]

        plot_images = {
            'Original images' : ben_image,
            'Advirsarial images' : adv_image,
            'Reconstructed images' : cln_image,
            'Originial Noise' : org_noise,
            'Advirsarial Noise' : adv_noise,
            'Reconstructed Noise' : adv_noise,
        }

        for indx, (title, data_image) in enumerate(plot_images.items()):
            ax = plt.subplot(len(plot_images), n, int(i+1+indx*n))
            plt.imshow(data_image, cmap='gist_gray', vmin=0, vmax=1)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
            if i == n//2:
                ax.set_title(title)

    plt.tight_layout()
    fig_dir = Path('{}/plots/{}_detection/images_{}_{:.3f}.jpg'.format(
        cfg.results_dir, attack_type, env, epsilon), dpi = 350)
    fig_dir.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_dir)
    plt.show() 

    # Plot the confidences
    plt.figure(figsize=(8,6))
    for i in range(n):
        ben_img_conf = conf_dict["ben_img_conf"][i]
        adv_img_conf = conf_dict["adv_img_conf"][i]
        cln_img_conf = conf_dict["cln_img_conf"][i]
        org_noi_conf = conf_dict["org_noi_conf"][i]
        adv_prt_conf = conf_dict["adv_prt_conf"][i]
        adv_noi_conf = conf_dict["adv_noi_conf"][i]

        plot_confs = {
            'Confidences on Original images' : ben_img_conf,
            'Confidences on Advirsarial images' : adv_img_conf,
            'Confidences on Reconstructed images' : cln_img_conf,
            'Confidences on Originial Noise' : org_noi_conf,
            'Confidences on Advirsarial Noise' : adv_prt_conf,
            'Confidences on Reconstructed Noise' : adv_noi_conf,
        }
        for indx, (title, data_conf) in enumerate(plot_confs.items()):
            ax = plt.subplot(len(plot_confs), n, int(i+1+indx*n))
            plt.bar(np.arange(len(data_conf)), data_conf)
            plt.yticks([0, 0.25, 0.5, 0.75, 1])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
            if i == n//2:
                ax.set_title(title)
            indx +=1

    plt.tight_layout()
    fig_dir = Path('{}/plots/{}_detection/confidences_{}_{:.3f}.jpg'.format(
        cfg.results_dir, attack_type, env, epsilon), dpi = 350)
    fig_dir.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_dir)
    plt.show() 

def filter_dataset_by_evaltype(cfg, results_df, col_dict):
    org_targets_col = 'Target'
    # nat_img_preds_col = col_dict['Natural_Org_Image']['Pred'][0]
    org_img_preds_col = col_dict['Original_Org_Image']['Pred'][0]
    adv_img_preds_col = col_dict['Adversarial_Org_Image']['Pred'][0]

    if cfg.evaltype == 0: # All
        clean_results = results_df
    elif cfg.evaltype == 1: # Successful attacks only
        filter_1 = results_df[org_targets_col] == results_df[org_img_preds_col] #ben_cls
        filter_2 = results_df[adv_img_preds_col] != results_df[org_img_preds_col] #!adv_cls
        # filter_3 = results_df['Epsilon'] == 0.0 #!adv_cls
        filter_3 = results_df['Attack'] == 'RN' #!adv_cls
        clean_results = results_df.where((filter_1 & filter_2) | filter_3).dropna()
    elif cfg.evaltype == 2: # Unsuccessful attacks only
        filter_1 = results_df[org_targets_col] == results_df[org_img_preds_col] #ben_cls
        filter_2 = results_df[adv_img_preds_col] == results_df[org_img_preds_col] #!adv_cls
        # filter_3 = results_df['Epsilon'] == 0.0 #!adv_cls
        filter_3 = results_df['Attack'] == 'RN'
        clean_results = results_df.where((filter_1 & filter_2) | filter_3).dropna()
    elif cfg.evaltype == 3: # Single class defense
        filter_1 = results_df[org_targets_col] == cfg.target_class #ben_cls
        clean_results = results_df.where(filter_1).dropna()
    return clean_results


def get_index_by_evaltype(cfg, results_df_org, attack):

    org_targets_col = 'Target'
    # nat_img_preds_col = col_dict['Natural_Org_Image']['Pred'][0]
    org_img_preds_col = 'Original_Org_Image_Pred'
    adv_img_preds_col = 'Adversarial_Org_Image_Pred'
    
    
    results_df = results_df_org[['Target', 'Original_Org_Image_Pred', 'Adversarial_Org_Image_Pred', 'Attack']]
    # print("Initial dim of results_df: ", results_df.shape)
    # print(results_df.head())

    # filter_1 = results_df['Epsilon'] == epsilon
    filter_2 = results_df['Attack'] == attack
    # results_df = results_df.where(filter_1 & filter_2).dropna()
    results_df = results_df.where(filter_2).dropna()
    # print("Cleaned dim of results_df: ", results_df.shape)
    # print(results_df.head())

    results_df = results_df.reset_index(drop = True)
    # print("New Index dim of results_df: ", results_df.shape)
    # print(results_df.head())


    if cfg.evaltype == 0: # All
        results_df = results_df
    elif cfg.evaltype == 1: # Successful attacks only
        filter_1 = results_df[org_targets_col] == results_df[org_img_preds_col] #ben_cls
        filter_2 = results_df[adv_img_preds_col] != results_df[org_img_preds_col] #!adv_cls
        results_df = results_df.where(filter_1 & filter_2).dropna()
    elif cfg.evaltype == 2: # Unsuccessful attacks only
        filter_1 = results_df[org_targets_col] == results_df[org_img_preds_col] #ben_cls
        filter_2 = results_df[adv_img_preds_col] == results_df[org_img_preds_col] #!adv_cls
        results_df = results_df.where(filter_1 & filter_2).dropna()
    elif cfg.evaltype == 3: # Single class defense
        filter_1 = results_df[org_targets_col] == cfg.target_class #ben_cls
        results_df = results_df.where(filter_1).dropna()

    # print("Final dim of results_df: ", results_df.shape)
    # print(results_df.head())

    return list(results_df.index)

# Source: https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py

term_width = 80

TOTAL_BAR_LENGTH = 10.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def poison_data(cfg, images, epsilon):
    adv_images = copy.deepcopy(images)
    count, channels, width, height = adv_images.shape
    max_value = torch.max(adv_images).item()
    for idx in range(count):
        # adv_images[idx] = add_trigger_to_image(images[idx], value=max_value, pos=3,) #value=max_value*epsilon/0.1
        adv_images[idx] = add_trigger_to_image(cfg, images[idx], color='yellow', value=max_value, pos=3, trigger_size=2)
    adv_images = torch.Tensor(adv_images)
    return adv_images

def add_trigger_to_image(cfg, image, color='yellow', value=255, pos=5, trigger_size=4):
    if cfg.dataset.name in ['fashion']:
        return add_trigger_to_image_v2(cfg, image, color, value, pos, trigger_size)
    else:
        return add_trigger_to_image_v1(cfg, image, color, value, pos, trigger_size)


def add_trigger_to_image_v1(cfg, image, color='yellow', value=255, pos=5, trigger_size=4):
    adv_image = copy.deepcopy(image)
    channel, width, height = adv_image.shape
    half_trigger_size = trigger_size // 2
    if value <= 1:
        value = 1.0
    else:
        value = 255
    if color == 'yellow':
        values = (value, value, 0)  # RGB value for yellow
    else:
        # Default to white if color is not specified or unknown
        values = (value, value, value)
    for c in range(channel):
        for i in range(-half_trigger_size, half_trigger_size + 1):  # Looping over trigger width
            for j in range(-half_trigger_size, half_trigger_size + 1):  # Looping over trigger height
                adv_image[c, width-(pos + i), height-(pos + j)] = values[c]
    return adv_image

def add_trigger_to_image_v2(cfg, image, color='yellow', value=255, pos=1, trigger_size=3, trigger_type='square'):
    """
    # Example usage:
    import numpy as np

    # Generate a random RGB image (for demonstration)
    image = np.random.randint(0, 256, (3, 28, 28), dtype=np.uint8)

    # Add trigger pattern (yellow square trigger of size 4 at position 5)
    trigger_image = add_trigger_to_image(image, color='yellow', pos=1, trigger_size=3, trigger_type='checkerboard')

    """
    adv_image = copy.deepcopy(image)
    channel, width, height = adv_image.shape
    half_trigger_size = trigger_size // 2
    
    if color == 'yellow':
        values = (value, value, 0)  # RGB value for yellow
    else:
        # Default to white if color is not specified or unknown
        values = (value, value, value)
    
    # Determine pattern position based on pos and trigger size
    pattern_position = (width - (pos + half_trigger_size + 1), height - (pos + half_trigger_size + 1))

    # Draw trigger pattern based on type
    if trigger_type == 'square':
        for c in range(channel):
            for i in range(-half_trigger_size, half_trigger_size + 1):
                for j in range(-half_trigger_size, half_trigger_size + 1):
                    adv_image[c, pattern_position[0] + i, pattern_position[1] + j] = values[c]
    elif trigger_type == 'checkerboard':
        for c in range(channel):
            for i in range(-half_trigger_size, half_trigger_size + 1):
                for j in range(-half_trigger_size, half_trigger_size + 1):
                    if (i + j) % 2 == 0:
                        adv_image[c, pattern_position[0] + i, pattern_position[1] + j] = values[c]
    elif trigger_type == 'circle':
        dist_from_center = ((np.arange(width) - pattern_position[0])**2 + 
                            (np.arange(height)[:, np.newaxis] - pattern_position[1])**2)**0.5
        for c in range(channel):
            adv_image[c, dist_from_center <= half_trigger_size] = values[c]

    return adv_image

# def poison_data(images, epsilon):
#     adv_images = copy.deepcopy(images)
#     channels, width, height = adv_images.shape[1:]
#     max_value = torch.max(adv_images).item()
#     for c in range(channels):
#         adv_images[:, c, width-3:width-1, height-3:height-1] = max_value*epsilon/0.2
#     adv_images = torch.Tensor(adv_images)
#     return adv_images


class PoisonedDataset(Dataset):

    def __init__(self, cfg, dataset, trigger_label, portion=0.1, mode="train", max_val = 255, device="cuda", dataname="mnist"):
        
        
        data, targets, classes =  self.get_data_classes(cfg, dataset, dataname)
        reshaped_data = self.reshape(data, dataname, max_val)
        self.class_num = len(classes)
        self.classes = classes
        # self.device =torch.device(device)
        self.max_val = max_val
        self.dataname = dataname
        self.data, self.targets = self.add_trigger(cfg, reshaped_data, targets, trigger_label, portion, mode, max_val)
        self.channels, self.width, self.height = self.__shape_info__()

    def __getitem__(self, item):
        img = self.data[item]
        label_idx = self.targets[item]
        # print(self.class_num)
        label = np.zeros(self.class_num)
        label[label_idx] = 1 # 把num型的label变成10维列表。
        label = torch.Tensor(label) 
        # img = img.to(self.device)
        # label = label.to(self.device)
        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[1:]
    
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def modify_array_dimensions(self, arr):
        if arr.ndim == 3:
            return np.expand_dims(arr, axis=1)
        if arr.ndim == 4 and np.argmin(arr.shape[1:]) != 0:
            return arr.transpose([0, 3, 1, 2])
        return arr

    def reshape(self, data, dataname="mnist", max_val=255):
        print("Original Shape of the dataset: ", data.shape)
        data = self.modify_array_dimensions(data)
        if max_val == 255 and np.max(data) <= 1.00:
            data = data * 255.0
        elif max_val == 1 and np.max(data) > 1.00:
            data = data/255.0

        print("New Shape of the dataset: ", data.shape)
        return np.array(data)
    
    def get_data_classes(self, cfg, dataset, dataset_name):
        try:
            print("Working on Other Dataset")
            all_data = dataset.data
            all_labels = dataset.targets
            classes = dataset.classes
        except:
            print("Working on GTSRB/CANCER Dataset")
            all_data = []
            all_labels = []
            for index in range(len(dataset)):
                data, label = dataset[index]
                all_data.append(data.numpy())
                all_labels.append(label)
            all_data = np.array(all_data)
            all_labels = np.array(all_labels)
            classes = cfg.dataset.classes
    
        return all_data, all_labels, classes
            
    def norm(self, data):
        offset = np.mean(data, 0)
        scale  = np.std(data, 0).clip(min=1)
        return (data- offset) / scale

    def add_trigger(self, cfg, data, targets, trigger_label, portion, mode, max_val):
        print("## generate " + mode + " Bad Imgs")
        new_data = copy.deepcopy(data)
        new_targets = copy.deepcopy(targets)
        perm = np.random.permutation(len(new_data))[0: int(len(new_data) * portion)]
        for idx in perm: 
            new_targets[idx] = trigger_label
            # new_data[idx] = add_trigger_to_image(new_data[idx], value=max_val, pos=3) # MNIST
            new_data[idx] = add_trigger_to_image(cfg, new_data[idx], color='yellow', value=max_val, pos=3, trigger_size=2)
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(new_data)-len(perm), portion))
        return torch.Tensor(new_data), new_targets

def imshow(cfg, img, model_type):
    npimg = copy.deepcopy(img)
    if type(npimg) != np.ndarray:
        npimg = npimg.numpy()
    if npimg.shape[0] < 4:
        npimg = np.transpose(npimg, (1, 2, 0))
    if np.max(npimg) > 1.00:
        npimg = npimg/255
    plt.imshow(npimg)
    image_dir = file_path(f"{cfg.results_dir}/dataloader/data_loader_sample_{model_type}_{cfg.dataset.name}.jpg")
    plt.savefig(image_dir)
    plt.show()


def get_all_data_from_loader(cfg, test_loader):
    
    all_images = []
    all_labels = []

    for org_images, targets in test_loader:
        all_images.append(org_images.numpy())
        all_labels.append(targets.numpy())

    # Concatenate all batches into a single ndarray
    all_images = np.concatenate(all_images, axis=0)[0:int(5*cfg.n)]
    all_labels = np.concatenate(all_labels, axis=0)[0:int(5*cfg.n)]
    print("all_images.shape, all_labels.shape : ", all_images.shape, all_labels.shape)
    return all_images, all_labels



def get_X_Y(train_loader):
    """
    Extracts input features (X_train_org) and corresponding labels (Y_train_org)
    from a PyTorch DataLoader.

    :param train_loader: PyTorch DataLoader containing training data
    :return: Tuple of NumPy arrays (X_train_org, Y_train_org)
    """
    X_train_org = []
    Y_train_org = []
    
    for batch_idx, batch in enumerate(train_loader):
        X_batch, Y_batch = batch
        X_train_org.append(X_batch.numpy())  # Convert tensor to NumPy array
        Y_train_org.append(Y_batch.numpy())  # Convert tensor to NumPy array
        if batch_idx == 250:
            break
    X_train_org = np.concatenate(X_train_org, axis=0)
    Y_train_org = np.concatenate(Y_train_org, axis=0)
    
    return X_train_org, Y_train_org



def get_random_samples(cfg, data_loader, data_type = 'valid', n = 100):
    """
    Extracts input features (X_train_org) and corresponding labels (Y_train_org)
    from a PyTorch DataLoader and returns a random sample of n examples.

    :param train_loader: PyTorch DataLoader containing training data
    :param n: Number of random samples to return
    :return: Tuple of NumPy arrays (X_train_sample, Y_train_sample)
    """
    
    _, test_loader, val_loader, _ = data_loader(cfg, 'target')
    if data_type == 'valid':
        data_loader = val_loader
    elif data_type == 'test':
        data_loader = test_loader

    # Collect all data from the DataLoader
    X_all = []
    Y_all = []
    
    for batch in data_loader:
        X_batch, Y_batch = batch
        X_all.append(X_batch.numpy())  # Convert tensor to NumPy array
        Y_all.append(Y_batch.numpy())  # Convert tensor to NumPy array

    # Concatenate all batches into single arrays
    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)
    
    # Create indices for sampling
    num_samples = X_all.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # Select random n samples
    selected_indices = indices[:n]
    X_train_sample = X_all[selected_indices]
    Y_train_sample = Y_all[selected_indices]
    
    return X_train_sample, Y_train_sample


def get_reconstructed_image(cfg, encoder, decoder, image_noisy):
    if cfg.autoencoder_type == 'CAE':
        rec_img  = decoder(encoder(image_noisy))
    elif cfg.autoencoder_type == 'VAE':
        rec_img  = decoder(encoder(image_noisy)[0])
    return rec_img

def get_reconstructed_image_unet(cfg, unet, image_noisy):
    torch.cuda.empty_cache()
    rec_img = unet(image_noisy)
    return rec_img



def retain_high_error_regions_batch(original_imgs, reconstructed_imgs, percentile=95):
    # Calculate the absolute reconstruction error for each image in the batch
    error = torch.abs(original_imgs - reconstructed_imgs)
    
    # Compute the average error across the color channels (assuming RGB, channels dimension is 1)
    avg_error = error.mean(dim=1, keepdim=True)  # Shape: (batch_size, 1, height, width)
    
    # Determine the threshold based on the specified percentile for each image
    thresholds = torch.quantile(avg_error.view(avg_error.size(0), -1), percentile / 100.0, dim=1)
    thresholds = thresholds.view(-1, 1, 1, 1)  # Reshape to match dimensions for broadcasting
    
    # Create a mask where the average error is higher than the threshold
    mask = (avg_error > thresholds).float()  # Shape: (batch_size, 1, height, width)
    
    # Expand the mask to have the same number of channels as the original images
    mask = mask.expand(-1, 3, -1, -1)  # Shape: (batch_size, 3, height, width)
    
    # Retain the regions with high error and set the rest to 0
    result = original_imgs * mask
    
    return result


def get_recon_noise(cfg, X_train_org, encoder, decoder, device, batch_size=256, return_org = False, percentile=95):
    if isinstance(X_train_org, np.ndarray):
        X_train_org = torch.tensor(X_train_org, dtype=torch.float32)
    
    num_samples = X_train_org.shape[0]
    X_noise_org = torch.empty_like(X_train_org)
    
    for i in range(0, num_samples, batch_size):
        batch_X = X_train_org[i:i + batch_size]
        batch_X = batch_X.to(device)
        # batch_noise = batch_X - decoder(encoder(batch_X))
        recon_batch_X = get_reconstructed_image(cfg, encoder, decoder, batch_X)
        if return_org:
            batch_noise = retain_high_error_regions_batch(batch_X, recon_batch_X, percentile=percentile)
        else:
            batch_noise = batch_X - recon_batch_X

        X_noise_org[i:i + batch_size] = batch_noise
    
    return X_noise_org




def get_recon_noise_unet(cfg, X_train_org, unet, device, batch_size=16):
    if isinstance(X_train_org, np.ndarray):
        X_train_org = torch.tensor(X_train_org, dtype=torch.float32)
    
    num_samples = X_train_org.shape[0]
    X_noise_org = torch.empty_like(X_train_org)
    
    for i in range(0, num_samples, batch_size):
        batch_X = X_train_org[i:i + batch_size]
        batch_X, _ = add_noise(batch_X, cfg.dataset.noise_factor)
        batch_X = batch_X.to(device)
        batch_noise = batch_X - get_reconstructed_image_unet(cfg, unet, batch_X)
        X_noise_org[i:i + batch_size] = batch_noise
    return X_noise_org


def randomize_pixels(tensor):
    """
    Randomize the positions of each pixel in a 4D tensor of shape (N, H, W, C).
    Args:
    tensor (torch.Tensor): Input tensor of shape (N, H, W, C).
    Returns:
    torch.Tensor: Output tensor with randomized pixel positions, same shape as input.
    """
    # Check if the input tensor has 4 dimensions
    if tensor.dim() != 4:
        raise ValueError("Input tensor must have 4 dimensions (N, H, W, C)")
    # Get the shape of the input tensor
    N, H, W, C = tensor.shape
    # Flatten the image to shape (N, H*W, C)
    flat_image = tensor.view(N, -1, C)
    # Shuffle the pixels
    shuffled_flat_image = flat_image[torch.arange(N).unsqueeze(-1), torch.randperm(H * W)]
    # Reshape back to (N, H, W, C)
    shuffled_image = shuffled_flat_image.view(N, H, W, C)
    return shuffled_image
    # # Example usage:
    # input_tensor = torch.randn(1, 32, 32, 3)  # or any other 4D shape
    # output_tensor = randomize_pixels(input_tensor)
    # print(output_tensor.shape)  # Should print the same shape as input_tensor

def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def load_state_dict(model, model_dir):
    # Assuming `encoder` is the model instance
    if torch.cuda.is_available():
        state_dict = torch.load(model_dir)
    else:
        # Load on CPU
        state_dict = torch.load(model_dir, map_location=torch.device('cpu'))
    

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove prefix "module."
        new_state_dict[name] = v
    # Load the modified state_dict
    model.load_state_dict(new_state_dict)

    return model


def get_tsne_vis(cfg, attack, class1_images, class2_images, label_1 = "Benign", label_2 = "Malicious", dim = "Original"):
    # Reshape the data to (n1, 784) and (n2, 784)
    class1_images_reshaped = class1_images.reshape(class1_images.shape[0], -1)
    class2_images_reshaped = class2_images.reshape(class2_images.shape[0], -1)
    cls1_df = pd.DataFrame(class1_images_reshaped)
    cls2_df = pd.DataFrame(class2_images_reshaped)
    cls1_df['Type'] = label_1
    cls2_df['Type'] = label_2

    df = pd.concat([cls1_df, cls2_df], axis=0)
    print("df shape: ", df.shape)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=10.0, random_state=42)
    tsne_results = tsne.fit_transform(df.drop('Type', axis=1))

    # Add the t-SNE results to the DataFrame
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]

    # Plot the t-SNE results
    plt.figure(figsize=(3, 3))
    plt.scatter(df[df['Type'] == label_1]['tsne-2d-one'], df[df['Type'] == label_1]['tsne-2d-two'],  alpha=0.6, label = label_1)
    plt.scatter(df[df['Type'] == label_2]['tsne-2d-one'], df[df['Type'] == label_2]['tsne-2d-two'],  alpha=0.6, label = label_2)
    plt.title(f"{attack} Attack")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    fig_dir = Path(f"{cfg.results_dir}/plots/tsne_vis")
    fig_dir.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/tsne_vis_{attack}_{dim}_{cfg.blackbox}_{cfg.evaltype}.jpg", dpi = cfg.fig_dpi)
    plt.savefig(f"{fig_dir}/tsne_vis_{attack}_{dim}_{cfg.blackbox}_{cfg.evaltype}.pdf")
    plt.show()
    print("TSNE Saved! ", fig_dir)

def get_tsne_df(cfg, attack, class1_images, class2_images, label_1 = "Benign", label_2 = "Malicious", dim = "Original"):
    class1_images_reshaped = class1_images.reshape(class1_images.shape[0], -1)
    class2_images_reshaped = class2_images.reshape(class2_images.shape[0], -1)
    cls1_df = pd.DataFrame(class1_images_reshaped)
    cls2_df = pd.DataFrame(class2_images_reshaped)
    cls1_df['Type'] = label_1
    cls2_df['Type'] = label_2

    df = pd.concat([cls1_df, cls2_df], axis=0)
    print("df shape: ", df.shape)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    tsne_results = tsne.fit_transform(df.drop('Type', axis=1))

    # Add the t-SNE results to the DataFrame
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    return df


def plot_tsne(cfg, tsne_data, attack_type='adversarial'):
    label_1 = "Benign"
    label_2 = "Malicious"
    total_attacks = len(tsne_data.keys())
    fig, axes = plt.subplots(2, total_attacks+1, figsize = (12, 4))
    for att_index, (attack, attack_data) in enumerate(tsne_data.items()):
        print(att_index, attack)
        for dt_index, (data_type, type_data) in enumerate(attack_data.items()):
            print(dt_index, data_type)
            class1_images, class2_images = type_data
            df = get_tsne_df(cfg, attack, class1_images, class2_images, label_1 = "Benign", label_2 = "Malicious", dim = data_type)
            # Plot the t-SNE results
            axes[dt_index, att_index].scatter(df[df['Type'] == label_1]['tsne-2d-one'], df[df['Type'] == label_1]['tsne-2d-two'],  alpha=0.6, label = label_1)
            axes[dt_index, att_index].scatter(df[df['Type'] == label_2]['tsne-2d-one'], df[df['Type'] == label_2]['tsne-2d-two'],  alpha=0.6, label = label_2)
            axes[dt_index, att_index].set_title(f"{attack} Attack")
            # axes[att_index, dt_index].set_xlabel("Feature 1")
            # axes[att_index, dt_index].set_ylabel("Feature 2")
            # if ax == axes[-1]:
            #     ax.legend()
            # else:
            #     ax.legend([])
    fig_dir = Path(f"{cfg.results_dir}/plots/tsne_vis")
    fig_dir.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/all_tsne_vis_{attack_type}_{cfg.blackbox}.jpg", dpi = cfg.fig_dpi)
    plt.savefig(f"{fig_dir}/all_tsne_vis_{attack_type}_{cfg.blackbox}.pdf")
    plt.show()
    print("TSNE Saved! ", fig_dir)


def shuffle_pixels(data):
    """
    Randomly shuffle the pixel positions of each image in the dataset 
    while preserving the RGB (or channel) values.

    Parameters:
    data (numpy.ndarray): The dataset of images with shape (N, C, H, W), 
                          where N is the number of images, C is the number of channels, 
                          H is the height, and W is the width.

    Returns:
    numpy.ndarray: The dataset with shuffled pixel positions.
    """
    # Get the shape of the dataset
    N, C, H, W = data.shape
    
    # Flatten each image to shape (N, C, H * W)
    data_flattened = data.reshape(N, C, H * W)
    
    # Shuffle pixel positions for each image
    for i in range(N):
        # Create a permutation of the indices [0, H*W-1]
        indices = np.random.permutation(H * W)
        # Apply the permutation to the columns of the flattened image
        data_flattened[i] = data_flattened[i, :, indices].T
    
    # Reshape back to (N, C, H, W)
    data_shuffled = data_flattened.reshape(N, C, H, W)
    
    return data_shuffled
