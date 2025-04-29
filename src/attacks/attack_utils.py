# Generate attacks
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from torchattacks import GN, FGSM, BIM, PGD, CW, JSMA,  OnePixel, Square
import torchattacks
from art.attacks.evasion import CarliniL2Method, UniversalPerturbation
from art.estimators.classification import PyTorchClassifier
# from attacks.cw_kkew import *
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import mahalanobis, cdist

from helper import *
import json
import copy

from numpy.linalg import norm
from scipy.stats import entropy


def configure_access(cfg, attack_type, net_dict):
    if cfg.blackbox:
        access_type = 'Black-box'
        net_target = net_dict['target']
        net_surrogate = net_dict['surrogate']
        scale_factor = 1.0  # Default scale factor for black-box
    elif attack_type == 'BadNet':
        access_type = 'Backdoor'
        net_target = net_dict['badnet']
        net_surrogate = net_dict['badnet']
        scale_factor = 255.0
    else:
        access_type = 'White-box'
        net_target = net_dict['target']
        net_surrogate = net_dict['target']
        scale_factor = 1.0  # Default scale factor for white-box

    return access_type, net_target, net_surrogate, scale_factor


def define_art_model(cfg, attack_type, net_surrogate):
    if attack_type in ['C&W', 'UAP']:
        input_shape = (cfg.dataset.num_channels, cfg.dataset.width, cfg.dataset.height) 
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net_surrogate.parameters(), lr=0.001)
        model_art = PyTorchClassifier(
            model=net_surrogate,
            clip_values=(0, 1),
            loss=criterion,
            optimizer=optimizer,
            input_shape=input_shape,
            nb_classes=cfg.dataset.num_classes,
        )
        return model_art
    return None
    
def handle_uap_attack(cfg, test_loader, model_art):
    eps = cfg.dataset.atk_cfg['UAP']['eps']
    delta  = cfg.dataset.atk_cfg['UAP']['delta']
    max_iter = cfg.dataset.atk_cfg['UAP']['max_iter']
    attacker = cfg.dataset.atk_cfg['UAP']['attacker']
    org_all_images, org_all_targets = get_all_data_from_loader(cfg, test_loader)
    attack = UniversalPerturbation(
        model_art, 
        delta=delta,
        max_iter=max_iter,
        attacker=attacker,
        eps=eps,
    )
    
    adv_all_images = attack.generate(org_all_images, org_all_targets)
    adv_all_prtrbs = (adv_all_images - org_all_images) #*eps
    print("Org Min/max : ", org_all_images.min(), org_all_images.max())
    print("Adv Min/max : ", adv_all_images.min(), adv_all_images.max())
    print("Ptr Min/max : ", adv_all_prtrbs.min(), adv_all_prtrbs.max())
    pre_gen_atk_data = {
        "UAP": {
            "org_all_images": org_all_images,
            "adv_all_images": adv_all_images,
            "adv_all_prtrbs": adv_all_prtrbs,
            "adv_univ_prtrbs": attack.noise,
        }
    }
    return pre_gen_atk_data

def register_forward_hook(cfg, net_target, hook_fn):
    
    model_instance = net_target.module if cfg.DataParallel else net_target

    try:
        hook_handle = getattr(model_instance, 'fc_feat').register_forward_hook(hook_fn)
        print("fc_feat is available!")

    except AttributeError:
        print("fc_feat does not exist! Loading avgpool instead!")
        hook_handle = getattr(model_instance, 'avgpool').register_forward_hook(hook_fn)

    return hook_handle

def register_forward_hook_backdoor(cfg, net_target, hook_fn):

    try:
        hook_handle = getattr(net_target, 'fc_feats').register_forward_hook(hook_fn)
        print("fc_feats is available!")

    except AttributeError:
        print("fc_feats does not exist! Loading avgpool instead!")
        hook_handle = getattr(net_target, 'avgpool').register_forward_hook(hook_fn)
    return hook_handle


def get_attack_params_from_yaml(cfg, attack_name):
    """
    Retrieve parameters for a specified attack from a YAML configuration file.
    
    Parameters:
        cfg (str): Yaml configuration file.
        attack_name (str): The name of the attack to retrieve parameters for.
        
    Returns:
        tuple: Values for all possible parameters (missing ones are None).
    """
    # Define all possible parameters across attacks
    possible_params = [
        "eps", "alpha", "steps", "theta", "gamma",
        "confidence", "max_iter", "binary_search_step",
        "learning_rate", "trigger_label", "poison_ratio",
        "x_dimensions", "y_dimensions", "n_queries", "p_init",
    ]
    
    # Get parameters for the specified attack, or an empty dict if not found
    attack_params = cfg.dataset.atk_cfg.get(attack_name, {})
    # Extract values for each possible parameter (default to None)
    return tuple(attack_params.get(param, "None") for param in possible_params)
    

def generate_attack(cfg, idx, attack_type, model, images, labels): #0.20 theta=1, gamma=0.1,

    eps, alpha, steps, theta, gamma, confidence, \
        max_iter, binary_search_step, learning_rate, \
            trigger_label, poison_ratio, \
                x_dimensions, y_dimensions, \
                    n_queries, p_init = get_attack_params_from_yaml(cfg, attack_type)
    
    # print(f"Attack: {attack_type}")
    # print(eps, alpha, steps, theta, gamma, confidence, 
    #       max_iter, binary_search_step, learning_rate,
    #       trigger_label, poison_ratio)

    device = torch.device(cfg.device)

    if len(labels.size()) > 1:
        labels = torch.argmax(labels.data, 1)
    if attack_type == 'GN':
        attack  = GN(model, std=0.0)
    if attack_type == 'FGSM': 
        attack = FGSM(model, eps=eps)
    if attack_type == 'BIM': 
        attack = BIM(model, eps=eps, alpha=alpha, steps=steps) 
    if attack_type == 'PGD': 
        attack = PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=True) 
    if attack_type == 'OnePixel':
        attack = OnePixel(model, pixels=1, steps=steps, popsize=10, inf_batch=128)
    if attack_type == 'Square':
        attack = Square(model, norm="Linf", eps=eps, n_queries=n_queries, n_restarts=1, p_init=p_init, loss="margin", resc_schedule=True,)
    if attack_type == 'JSMA':         
        attack = JSMA(model, theta=theta, gamma=gamma)   
    
    # ART library------------
    if attack_type == 'C&W':
        attack = CarliniL2Method(model, targeted=False, 
                                max_iter=max_iter, 
                                binary_search_steps=binary_search_step, 
                                confidence = confidence, 
                                learning_rate=learning_rate, 
                                verbose=False) 
    
        adv_images = attack.generate(images.detach().cpu().numpy())
        adv_images = torch.tensor(adv_images)
        adv_images = adv_images.to(device)
        adv_noises = torch.tensor(adv_images - images)
        return adv_images, adv_noises
        
    if attack_type == 'UAP': 
        org_images = torch.tensor(images["UAP"]['org_all_images'][idx])
        adv_noises = torch.tensor(images["UAP"]['adv_univ_prtrbs']) 
        adv_images = (org_images + adv_noises).to(device)
        adv_noises = adv_noises.to(device)
        return adv_images, adv_noises
    
    # Added library--------------
    if attack_type == 'RN':
        adv_noises = eps * torch.randn_like(images)
        adv_images = images + adv_noises
        adv_images = torch.clamp(adv_images, min=0, max=1)
        adv_noises = adv_noises.to(device)
        adv_noises = adv_noises.to(device)
        return adv_images, adv_noises

    if attack_type == 'BadNet':
        adv_images = poison_data(cfg, images, eps)
        device = torch.device(cfg.device)
        adv_images = adv_images.to(device)
        adv_noises = torch.tensor(adv_images - images)
        return adv_images, adv_noises
    
    adv_images = attack(images, labels)
    adv_noises = (adv_images - images)

    return adv_images, adv_noises


def hook_fn(module, input, output):
        global inter_feat
        inter_feat = output

def softmax(x):
    exp_x = np.exp(x)  
    return exp_x / exp_x.sum(axis=0, keepdims=True)


import numpy as np
from scipy.stats import entropy
from numpy.linalg import norm

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)

def magnet_JSD(P, Q, softmax_done=False, T=1):
    # Test case 1: Assert that P and Q are not empty
    assert len(P) > 0 and len(Q) > 0, "P and Q must not be empty"

    # Test case 2: Assert that P and Q have the same length
    assert len(P) == len(Q), "P and Q must have the same length"

    # Test case 3: Assert that P and Q are finite (not NaN or Inf)
    assert np.all(np.isfinite(P)) and np.all(np.isfinite(Q)), "P and Q must be finite"

    if not softmax_done:
        P = softmax(P / T)
        Q = softmax(Q / T)
        
    epsilon = 1e-10
    P = P + epsilon
    Q = Q + epsilon
    
    # Normalize P and Q
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)

    # Test case 4: Assert that normalization did not result in NaN
    assert np.all(np.isfinite(_P)) and np.all(np.isfinite(_Q)), "Normalization resulted in NaN"

    # Compute M
    _M = 0.5 * (_P + _Q)

    # Test case 5: Assert that _M contains no zeros to avoid log(0)
    assert np.all(_M > 0), f"P: {_P}, Q: {_Q}, M: {_M} \n_M contains zeros, which would result in log(0)"

    # Compute the Jensen-Shannon Divergence score
    score = 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

    # Test case 6: Check for infinity or NaN in the final score
    if not np.isfinite(score):
        # print("Score is not finite. P:", P, "Q:", Q, "Score:", score)
        raise ValueError("Score resulted in NaN or Inf")

    return score


def magnet_Norm(raw_img, rec_img, norm='L1'):
    if norm == 'L1':
        return np.sum(np.abs(raw_img - rec_img))
    # Add other normalization options here if needed
    else:
        raise ValueError("Invalid normalization type. Supported types: 'L1'.")
    
def magnet_attributes(img_conf_pre_feat):
    magnet_nat_l1 = magnet_Norm(img_conf_pre_feat["Natural_Org_Image"]["Image"], 
                                img_conf_pre_feat["Natural_Rec_Image"]["Image"], 
                                norm='L1')
    magnet_ben_l1 = magnet_Norm(img_conf_pre_feat["Benign_Org_Image"]["Image"], 
                                img_conf_pre_feat["Benign_Rec_Image"]["Image"], 
                                norm='L1')
    magnet_adv_l1 = magnet_Norm(img_conf_pre_feat["Adversarial_Org_Image"]["Image"], 
                                img_conf_pre_feat["Adversarial_Rec_Image"]["Image"], 
                                norm='L1')
    
    magmet_nat_jds = magnet_JSD(img_conf_pre_feat["Natural_Org_Image"]["Conf"], 
                                img_conf_pre_feat["Natural_Rec_Image"]["Conf"],
                                softmax_done = False)
    magmet_ben_jds = magnet_JSD(img_conf_pre_feat["Benign_Org_Image"]["Conf"], 
                                img_conf_pre_feat["Benign_Rec_Image"]["Conf"], 
                                softmax_done = False)
    magmet_adv_jds = magnet_JSD(img_conf_pre_feat["Adversarial_Org_Image"]["Conf"], 
                                img_conf_pre_feat["Adversarial_Rec_Image"]["Conf"], 
                                softmax_done = False)

    return [magnet_nat_l1, magnet_ben_l1, magnet_adv_l1, magmet_nat_jds, magmet_ben_jds, magmet_adv_jds]



def get_feature_rep(cfg, rec_noise, net, scale_factor=1.0):
    X_feat_list = []
    y_pred_list = []
    for rec_noise_sample in rec_noise:
        X_feat_list.append(get_model_output(cfg, rec_noise_sample, net, scale_factor=1.0)['Feat'])
        y_pred_list.append(get_model_output(cfg, rec_noise_sample, net, scale_factor=1.0)['Pred'])
    X_feat = np.stack(X_feat_list)
    y_pred = np.stack(y_pred_list)
    return X_feat, y_pred


def get_model_output(cfg, image, model, scale_factor=1.0, return_dist = False, repeat=50):

    feat_dist = []
    with torch.no_grad():
        if image.ndim < 4:
            image = image.unsqueeze(0)
            # image = np.expand_dims(image, axis=0)

        if cfg.scale:
            image = scale_to_0_1(image)
        
        conf = model((image * scale_factor).float())
        pred = conf.max(1, keepdim=True)[1].detach().cpu().item()
        conf = conf.squeeze().detach().cpu().numpy() 
        feat = inter_feat.clone().squeeze().detach().cpu().numpy()
        norm = torch.norm(image).detach().cpu().item()
        
        # Randomize and generate a dist of features
        if return_dist:
            _image = copy.copy(image)
            feat_dist = []
            for indx in range(repeat):
                _image = randomize_pixels(_image)
                _conf = model((_image * scale_factor).float())
                _feat = inter_feat.clone().squeeze().detach().cpu().numpy()
                feat_dist.append(_feat)
        
            feat_dist = np.vstack(feat_dist)
        image = image.squeeze().detach().cpu().numpy()
    
    out = {"Image": image, "Conf" : conf, "Pred" : pred, "Feat" : feat, "Norm": norm, "Dist" : feat_dist}
    return out 

def check_temp(cfg, accessType, attack_type, attack_ext, env, num_of_samples):
        
    ext = f"{cfg.dataset.name}_{accessType}_{attack_type}_{attack_ext}_{env}_{num_of_samples}"
    data_dir = Path(f"{cfg.results_dir}/temp/attack_analysis_df_{ext}.csv")
    dict_dir = Path(f"{cfg.results_dir}/temp/attack_accuracy_{ext}.json")

    if data_dir.is_file() and dict_dir.is_file() and (not cfg.reeval):
        attack_analysis_df = pd.read_csv(data_dir, index_col=0)
        with open(dict_dir, 'r') as file:
            accuracy_dict = json.load(file)
        results_dict = {
            'attack_analysis': attack_analysis_df,
            'attack_accuracy': accuracy_dict,
            }
        print("Existing evaluation data loaded from temp................")
        return results_dict
    else:
        print("No existing evaluation data.....!!!")
        print(data_dir)
        print(dict_dir)
        return None


# Define a function to apply the transformations
def apply_real_world_transform(cfg, tensor):
    device = torch.device(cfg.device)
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color changes
        # transforms.GaussianBlur(kernel_size=3),  # Blur effect
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),  # Slight rotation and translation
        # transforms.RandomPerspective(distortion_scale=0.25, p=1.0),  # Perspective distortion
    ])

    # Convert tensor to PIL Image based on channel count
    pil_image = transforms.ToPILImage()(tensor.squeeze(0))
    # Apply transformations
    transformed_image = transform(pil_image)
    # Convert back to tensor
    transformed_tensor = transforms.ToTensor()(transformed_image)
    transformed_tensor = transformed_tensor.unsqueeze(0).to(device) # Add back the batch dimension
    transformed_tensor.to(device)
    return transformed_tensor
