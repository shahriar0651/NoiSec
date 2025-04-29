# Source: https://github.com/ning-wang1/manda/blob/344d0d20f5d56830de3936d2b06917a533ca4304/utils/util_artifact.py#L225

import torch
import numpy as np
from  pathlib import Path
from sklearn.semi_supervised import LabelSpreading
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

# Actual imports
import os
import argparse
import warnings
import numpy as np
from sklearn.neighbors import KernelDensity

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset

import multiprocessing as mp
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def normalize(normal, adv, noisy):
    """
    TODO
    :param normal:
    :param adv:
    :param noisy:
    :return:
    """
    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv, noisy)))

    return total[:n_samples], total[n_samples:2*n_samples], total[2*n_samples:]

def score_point(tup):
    """
    TODO
    :param tup:
    :return:
    """
    x, kde = tup

    return kde.score_samples(np.reshape(x, (1, -1)))[0]

def score_samples(kdes, samples, preds, n_jobs=None):
    """
    TODO
    :param kdes:
    :param samples:
    :param preds:
    :param n_jobs:
    :return:
    """
    if n_jobs is not None:
        p = mp.Pool(n_jobs)
    else:
        p = mp.Pool()
    results = np.asarray(
        p.map(
            score_point,
            [(x, kdes[i]) for x, i in zip(samples, preds)]
        )
    )
    p.close()
    p.join()

    return results

def train_lr(densities_pos, densities_neg, uncerts_pos, uncerts_neg):
    """
    TODO
    :param densities_pos:
    :param densities_neg:
    :param uncerts_pos:
    :param uncerts_neg:
    :return:
    """
    values_neg = np.concatenate(
        (densities_neg.reshape((1, -1)),
         uncerts_neg.reshape((1, -1))),
        axis=0).transpose([1, 0])
    values_pos = np.concatenate(
        (densities_pos.reshape((1, -1)),
         uncerts_pos.reshape((1, -1))),
        axis=0).transpose([1, 0])

    values = np.concatenate((values_neg, values_pos))
    labels = np.concatenate(
        (np.zeros_like(densities_neg), np.ones_like(densities_pos)))

    lr = LogisticRegressionCV(n_jobs=-1).fit(values, labels)

    return values, labels, lr

def compute_roc(probs_neg, probs_pos, plot=False):
    """
    TODO
    :param probs_neg:
    :param probs_pos:
    :param plot:
    :return:
    """
    probs = np.concatenate((probs_neg, probs_pos))
    labels = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score

dataset = 'mnist'
BANDWIDTHS = {'mnist': 1.20, 'cifar': 0.26, 'svhn': 1.00}



# def get_mc_predictions(model, X, nb_iter=50, batch_size=64):
#     """
#     Generates Monte Carlo predictions using a PyTorch model with dropout enabled during prediction.

#     :param model: PyTorch model
#     :param X: Input data (NumPy array or PyTorch tensor)
#     :param nb_iter: Number of Monte Carlo iterations (default: 50)
#     :param batch_size: Batch size for prediction (default: 256)
#     :return: NumPy array of predictions with shape (nb_iter, len(X), output_dim)
#     """
#     model.train()  # Set model to evaluation mode
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Convert input data to PyTorch tensor and move to appropriate device
#     if isinstance(X, np.ndarray):
#         X = torch.tensor(X, dtype=torch.float32)
#     X = X.to(device)
#     print("X Shape :", X.shape)
#     print("X Type: ", X.type())
    
#     output_dim = 10  # Determine output dimension
#     print("output_dim : ", output_dim)

#     # Function to make predictions with dropout enabled
#     def predict():
#         n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
#         output = np.zeros(shape=(len(X), output_dim))
#         with torch.no_grad():
#             for i in range(n_batches):
#                 start_idx = i * batch_size
#                 end_idx = min((i + 1) * batch_size, len(X))
#                 batch_X = X[start_idx:end_idx]
#                 output[start_idx:end_idx] = model(batch_X).cpu().numpy()
#         return output

#     preds_mc = []
#     for _ in tqdm(range(nb_iter)):
#         output = predict()
#         print(f"{_+1} ==>", np.mean(output), np.std(output))
#         preds_mc.append(output)
#     # print(preds_mc)
#     return np.asarray(preds_mc)


def get_mc_predictions(cfg, model, X, nb_iter=50, batch_size=64):
    """
    Generates Monte Carlo predictions using a PyTorch model with dropout enabled during prediction.

    :param model: PyTorch model
    :param X: Input data (NumPy array or PyTorch tensor)
    :param nb_iter: Number of Monte Carlo iterations (default: 50)
    :param batch_size: Batch size for prediction (default: 256)
    :return: NumPy array of predictions with shape (nb_iter, len(X), output_dim)
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.train()  # Set model to evaluation mode
    # print(model)
    # print("Model in training mode:", model.training)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert input data to PyTorch tensor and move to appropriate device
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    X = X.to(device)
    
    output_dim = cfg.dataset.num_classes  # Determine output dimension

    # Function to make predictions with dropout enabled
    def predict(model, X, batch_size, output_dim):
        n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        output = np.zeros(shape=(len(X), output_dim))
        # with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X))
            batch_X = X[start_idx:end_idx]
            output[start_idx:end_idx] = model(batch_X).detach().cpu().numpy()
        return output

    preds_mc = []
    for _ in tqdm(range(nb_iter)):
        output = predict(model, X, batch_size, output_dim)
        preds_mc.append(output)
    return np.asarray(preds_mc)


def get_deep_representations(cfg, model, X, layer_name='fc_feat', batch_size=256):
    """
    Extract deep representations from a specified hidden layer of a PyTorch model.

    :param model: Trained PyTorch model
    :param X: Input data (NumPy array or PyTorch tensor)
    :param layer_name: Name of the layer from which to extract representations
    :param batch_size: Batch size for processing (default: 256)
    :return: NumPy array of deep representations
    """
    device = next(model.parameters()).device  # Get device of model parameters
    # Convert input data to PyTorch tensor and move to appropriate device
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    X = X.to(device)

    # DataLoader for batching
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Hook to capture the output of the specified layer
    representations = []


    def hook(module, input, output):
        rep = np.squeeze(output.detach().cpu().numpy())
        if rep.ndim == 1:
            rep = rep.reshape(1, -1)
        representations.append(rep)

    model_instance = model.module if cfg.DataParallel else model
    try:
        hook_handle = getattr(model_instance, layer_name).register_forward_hook(hook)
    except:
        hook_handle = getattr(model_instance, 'avgpool').register_forward_hook(hook)
   
    # Forward pass to capture representations
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_batch = batch[0].to(device)
            model(input_batch)

    # Remove the hook
    hook_handle.remove()

    # Concatenate all batch representations
    representations = np.concatenate(representations, axis=0)
    return representations


def predict_classes(model, X, batch_size=32):
    """
    Predict classes for the given input data using the provided PyTorch model.
    
    :param model: PyTorch model
    :param X: Input data (NumPy array or PyTorch tensor)
    :param batch_size: Batch size for processing (default: 256)
    :return: NumPy array of predicted classes
    """
    model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device  # Get device of model parameters

    # Convert input data to PyTorch tensor and move to appropriate device
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    X = X.to(device)

    # DataLoader for batching
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_batch = batch[0].to(device)
            probs = model(input_batch)
            preds = probs.argmax(dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    return all_probs, all_preds



def get_accuracy(net, X, Y, batch_size=64):
    net.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    
    # Convert X and Y to tensors if they are numpy arrays
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if isinstance(Y, np.ndarray):
        Y = torch.tensor(Y, dtype=torch.long)
    
    device = next(net.parameters()).device  # Get device of model parameters
    
    # No need to track gradients for validation/testing
    with torch.no_grad():
        # Process data in batches
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            X_batch = X[start:end].to(device)
            Y_batch = Y[start:end].to(device)
            
            # Forward pass
            outputs = net(X_batch)
            
            # Get the predicted class with the highest score
            _, predicted = torch.max(outputs.data, 1)
            _, actual = torch.max(Y_batch.data, 1)
            # Total number of samples
            total += Y_batch.size(0)
            
            # Count the number of correct predictions
            # predicted = predicted.cpu().numpy()
            # actual = actual.cpu().numpy()
            # print(predicted, actual)
            correct_pred = (predicted == actual)

            # print(correct_pred)
            correct += correct_pred.sum().item()
            
            # Clear tensors from device
            X_batch = X_batch.cpu()
            Y_batch = Y_batch.cpu()
    
    # Calculate accuracy
    accuracy = correct / total
    return accuracy

def get_artifacts_score(cfg, net_target, X_train_org, Y_train_org, selected_indices, attack, env, file_path, scale_factor):

    # Load natural and adversarial images.....
    data_dir = file_path(f"{cfg.results_dir}/data/{attack}/attack_data_Natural_Org_Image_{env}_{cfg.blackbox}.npy")
    nat_images = np.load(data_dir) * scale_factor
    if nat_images.ndim == 3:
        nat_images = np.expand_dims(nat_images, axis=1)

    data_dir = file_path(f"{cfg.results_dir}/data/{attack}/attack_data_Benign_Org_Image_{env}_{cfg.blackbox}.npy")
    ben_images = np.load(data_dir) * scale_factor
    if ben_images.ndim == 3:
        ben_images = np.expand_dims(ben_images, axis=1)
    
    data_dir = file_path(f"{cfg.results_dir}/data/{attack}/attack_data_Adversarial_Org_Image_{env}_{cfg.blackbox}.npy")
    adv_images = np.load(data_dir) * scale_factor
    if adv_images.ndim == 3:
        adv_images = np.expand_dims(adv_images, axis=1)

    nat_images = nat_images[selected_indices]
    ben_images = ben_images[selected_indices]
    adv_images = adv_images[selected_indices]

    print(f"\n\n\n Benign and adversarial images: {ben_images.shape}, {adv_images.shape}\n\n")
    

    batch_size = 32 #TODO: Move to config file
    # print('Getting Monte Carlo dropout variance predictions...')
    # acc = get_accuracy(net_target, X_train_org, Y_train_org)
    # print(f"Accuracy on training data: {acc}")
                
    ## Get Bayesian uncertainty scores
    uncerts_normal = get_mc_predictions(cfg, net_target, nat_images, nb_iter=50,
                                        batch_size=batch_size).var(axis=0).mean(axis=1)
    uncerts_noisy = get_mc_predictions(cfg, net_target, ben_images, nb_iter=50,
                                        batch_size=batch_size).var(axis=0).mean(axis=1)
    uncerts_adv = get_mc_predictions(cfg, net_target, adv_images,nb_iter=50,
                                        batch_size=batch_size).var(axis=0).mean(axis=1)
    # print(uncerts_normal.mean(), uncerts_noisy.mean(), uncerts_adv.mean())

    ## Get KDE scores
    # Get deep feature representations
    print('Getting deep feature representations...')
    X_train_features = get_deep_representations(cfg, net_target, X_train_org, layer_name='fc_feat',
                                                batch_size=batch_size) 
    X_test_normal_features = get_deep_representations(cfg, net_target, nat_images, layer_name='fc_feat',
                                                        batch_size=batch_size)
    X_test_noisy_features = get_deep_representations(cfg, net_target, ben_images, layer_name='fc_feat',
                                                        batch_size=batch_size)
    X_test_adv_features = get_deep_representations(cfg, net_target, adv_images, layer_name='fc_feat',
                                                    batch_size=batch_size)
    
    
    # Train one KDE per class
    print('Training KDEs...')
    class_inds = {}
    for i in range(cfg.dataset.num_classes):
        # class_inds[i] = np.where(Y_train_org.argmax(axis=1) == i)[0] 
        class_inds[i] = np.where(Y_train_org == i)[0] 
    kdes = {}
    warnings.warn("Using pre-set kernel bandwidths that were determined "
                    "optimal for the specific CNN models of the paper. If you've "
                    "changed your model, you'll need to re-optimize the "
                    "bandwidth.")
    for i in tqdm(range(cfg.dataset.num_classes)):                        
        try:
            kdes[i] = KernelDensity(kernel='gaussian',
                                bandwidth=cfg.dataset.bandwidth) \
            .fit(X_train_features[class_inds[i]])
        except:
            #kdes[i] = KernelDensity(kernel='gaussian', bandwidth=cfg.dataset.bandwidth)
            # kdes[i] = kdes[i-1] #FIXME: Temp solution: Skip if class do not have any voice command
            kdes[i] = KernelDensity(kernel='gaussian',
                                bandwidth=cfg.dataset.bandwidth) \
            .fit(X_train_features)
    # Get model predictions
    print('Computing model predictions...')
    _ , preds_test_normal = predict_classes(net_target, nat_images, batch_size=batch_size)
    _ , preds_test_noisy = predict_classes(net_target, ben_images, batch_size=batch_size)
    _ , preds_test_adv = predict_classes(net_target, adv_images, batch_size=batch_size)

    # Get density estimates
    print('computing densities...')
    densities_normal = score_samples(
        kdes,
        X_test_normal_features,
        preds_test_normal
    )
    densities_noisy = score_samples(
        kdes,
        X_test_noisy_features,
        preds_test_noisy
    )
    densities_adv = score_samples(
        kdes,
        X_test_adv_features,
        preds_test_adv
    )

    ## Z-score the uncertainty and density values
    uncerts_normal_z, uncerts_adv_z, uncerts_noisy_z = normalize(
        uncerts_normal,
        uncerts_adv,
        uncerts_noisy
    )
    densities_normal_z, densities_adv_z, densities_noisy_z = normalize(
        densities_normal,
        densities_adv,
        densities_noisy
    )

    ## Build detector
    values, labels, lr = train_lr(
        densities_pos=densities_adv_z,
        densities_neg=np.concatenate((densities_normal_z, densities_noisy_z)),
        uncerts_pos=uncerts_adv_z,
        uncerts_neg=np.concatenate((uncerts_normal_z, uncerts_noisy_z))
    )
    print(values.shape)

    ## Evaluate detector
    # Compute logistic regression model predictions
    probs = lr.predict_proba(values)[:, 1]
    # Compute AUC
    n_samples = len(adv_images)
    # The first 2/3 of 'probs' is the negative class (normal and noisy samples),
    # and the last 1/3 is the positive class (adversarial samples).
    _, _, auc_score = compute_roc(
        probs_neg=probs[:2 * n_samples],
        probs_pos=probs[2 * n_samples:]
    )
    # print(probs[:n_samples])
    # print(probs[n_samples:2*n_samples])
    # print(probs[2 * n_samples:])

    print('1. Detector ROC-AUC score: %0.4f\n\n' % auc_score)


    return probs[n_samples:]



def generate_noisy_versions(x, sigma=0.005, num_of_noisy_samples=3):
    """
    Generates multiple noisy versions of the input `x`, where the noise is drawn from a normal distribution
    with a standard deviation of `sigma`. The noisy values are clamped within the range [0, 1].
    
    Parameters:
    x (np.array): The original input data.
    sigma (float): The standard deviation of the Gaussian noise.
    modify_variants (int): The number of noisy variants to generate.
    
    Returns:
    np.array: A concatenated array of noisy versions of `x`, with values clipped between 0 and 1.
    """
    # Generate the first noisy version
    if x.ndim == 2:
        x = np.expand_dims(x, axis=(0, 1))
    elif x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    x_noisy = x.copy()

    # Generate additional noisy variants and concatenate them
    for _ in range(num_of_noisy_samples):
        x_new = np.clip(x + np.random.normal(loc=0, scale=sigma, size=x.shape), 0, 1)
        x_noisy = np.concatenate([x_noisy, x_new])
    return x_noisy


def uncertainty(y):
    """
    Computes the uncertainty as the difference between the sum of norms of individual vectors
    and the norm of the summed vector, scaled by the length of the input array.
    
    Parameters:
    y (np.array): A 2D array where each row represents a vector.
    
    Returns:
    float: The uncertainty value.
    """
    # Compute the norms of each individual vector (rows of y)
    norm_sum = np.linalg.norm(y, axis=1).sum()
    
    # Compute the norm of the sum of all vectors
    sum_norm = np.linalg.norm(np.sum(y, axis=0))
    
    # Compute the uncertainty difference and scale by the number of vectors
    diff = (1 / len(y)) * (norm_sum - sum_norm)
    
    return diff

def print_feature_specs(feats, feat_name):
    print(f"\n--- {feat_name} Specifications ---")
    print(f"Shape: {feats.shape}")
    print(f"Data Type: {feats.dtype}")
    print(f"Memory Usage (MB): {feats.nbytes / (1024 ** 2):.2f} MB")
    print(f"Mean: {np.mean(feats):.4f}")
    print(f"Standard Deviation: {np.std(feats):.4f}")
    print(f"Min value: {np.min(feats):.4f}")
    print(f"Max value: {np.max(feats):.4f}")
    print(f"Norm (L2): {np.linalg.norm(feats):.4f}")


def get_manda_score(cfg, net_target, X_train_org, Y_train_org, selected_indices, attack, env, file_path, scale_factor):
    # print("Starting Manda")
    # print(X_train_org.shape, Y_train_org.shape)
    print("Starting Manda")
    # Load natural and adversarial images.....
    data_dir = file_path(f"{cfg.results_dir}/data/{attack}/attack_data_Natural_Org_Image_{env}_{cfg.blackbox}.npy")
    nat_images = np.load(data_dir) * scale_factor

    
    # nat_images = X_train_org # Replacing with X_train_org


    if nat_images.ndim == 3:
        nat_images = np.expand_dims(nat_images, axis=1)

    data_dir = file_path(f"{cfg.results_dir}/data/{attack}/attack_data_Benign_Org_Image_{env}_{cfg.blackbox}.npy")
    ben_images = np.load(data_dir) * scale_factor
    if ben_images.ndim == 3:
        ben_images = np.expand_dims(ben_images, axis=1)
    
    data_dir = file_path(f"{cfg.results_dir}/data/{attack}/attack_data_Adversarial_Org_Image_{env}_{cfg.blackbox}.npy")
    adv_images = np.load(data_dir) * scale_factor
    if adv_images.ndim == 3:
        adv_images = np.expand_dims(adv_images, axis=1)

    nat_images = nat_images[selected_indices]
    ben_images = ben_images[selected_indices]
    adv_images = adv_images[selected_indices]

    nat_labels = Y_train_org[selected_indices]

    # print(f"\n\n\n Benign and adversarial images: {ben_images.shape}, {adv_images.shape}\n\n")
    

    batch_size = 32 #TODO: Move to config file

    # Get deep feature representations
    # print('Getting deep feature representations...')
    feats_train = get_deep_representations(cfg, net_target, X_train_org, layer_name='fc_feat',
                                                batch_size=batch_size) 
    
    feats_nat = get_deep_representations(cfg, net_target, nat_images, layer_name='fc_feat',
                                                        batch_size=batch_size)
    feats_ben = get_deep_representations(cfg, net_target, ben_images, layer_name='fc_feat',
                                                        batch_size=batch_size)
    feats_adv = get_deep_representations(cfg, net_target, adv_images, layer_name='fc_feat',
                                                    batch_size=batch_size)


    # # Print specifications and compare feature sets
    # print_feature_specs(feats_train, "feats_train")
    # print_feature_specs(feats_nat, "feats_nat")
    # print_feature_specs(feats_ben, "feats_ben")
    # print_feature_specs(feats_adv, "feats_adv")

    # print('Training Label Spreading...')
    # consistency_model = LabelSpreading(gamma=20)
    consistency_model = LabelSpreading(kernel='knn', n_neighbors=5)  # Instead of 'rbf'

    # print(feats_train.shape, Y_train_org.shape)
    consistency_model.fit(feats_train, np.argmax(Y_train_org, axis=1))

    # print('Training Label Spreading...')
    # consistency_model = LabelSpreading(gamma=6)
    # print(feats_nat.shape, nat_labels.shape)
    # consistency_model.fit(feats_nat, np.argmax(nat_labels, axis=1))
    # print(consistency_model.predict(feats_nat))

    # Start predicting 
    # print('Target model predictions...')
    probs_cls_nat, preds_cls_nat = predict_classes(net_target, nat_images, batch_size=batch_size)
    probs_cls_ben, preds_cls_ben = predict_classes(net_target, ben_images, batch_size=batch_size)
    probs_cls_adv, preds_cls_adv = predict_classes(net_target, adv_images, batch_size=batch_size)


    # print('LabelSpreading model predictions...')
    preds_lbs_train = consistency_model.predict(feats_train)
    preds_lbs_nat = consistency_model.predict(feats_nat)
    preds_lbs_ben = consistency_model.predict(feats_ben)
    preds_lbs_adv = consistency_model.predict(feats_adv)




    # print("Original labels of train: ", np.argmax(Y_train_org, axis=1))
    # print("Predicted labels of train: ", preds_lbs_train)

    # print("Predicted natural classes: ", preds_cls_nat)
    # print("Predicted natural labels: ", preds_lbs_nat)

    # print("Predicted benign classes: ", preds_cls_ben)
    # print("Predicted benign labels: ", preds_lbs_ben)

    # print("Predicted adversarial classes: ", preds_cls_adv)
    # print("Predicted adversarial labels: ", preds_lbs_adv)

    # print(preds_cls_ben.shape, preds_cls_adv.shape)

    all_cls_preds = np.concatenate([preds_cls_ben, preds_cls_adv])
    all_lbs_preds = np.concatenate([preds_lbs_ben, preds_lbs_adv])
    all_cls_probs = np.concatenate([probs_cls_ben, probs_cls_adv])
    all_feats = np.concatenate([feats_ben, feats_adv])
    all_images = np.concatenate([ben_images, adv_images])
    attack_prob = []
    for pred_cls, pred_lbs, prob_cls, feats, image in zip(all_cls_preds, all_lbs_preds, all_cls_probs, all_feats, all_images):
        if not pred_cls == pred_lbs:
            attack_prob.append(5.0) #High confidence attack
        else:
            image_noisy = generate_noisy_versions(image, sigma=0.005, num_of_noisy_samples=3)
            probs_cls_noisy, _ = predict_classes(net_target, image_noisy, batch_size=batch_size)
            attack_score = uncertainty(probs_cls_noisy)
            attack_prob.append(attack_score)
    
    attack_prob = np.array(attack_prob)
    n = attack_prob.shape[0]//2
    mean_benign, std_benign = np.mean(attack_prob[0:n]), np.std(attack_prob[0:n])
    mean_advers, std_advers = np.mean(attack_prob[n:]), np.std(attack_prob[n:])

    # print("\n\nBenign scores: ", attack_prob[0:n])
    # print("\n\nAdvers scores: ", attack_prob[n:])
    
    # print("mean_benign, std_benign : ", mean_benign, std_benign)
    # print("mean_advers, std_advers : ", mean_advers, std_advers)
    
    return attack_prob

    # class_inds = {}
    # for i in range(cfg.dataset.num_classes):
    #     # class_inds[i] = np.where(Y_train_org.argmax(axis=1) == i)[0] 
    #     class_inds[i] = np.where(Y_train_org == i)[0] 
    # kdes = {}
    # warnings.warn("Using pre-set kernel bandwidths that were determined "
    #                 "optimal for the specific CNN models of the paper. If you've "
    #                 "changed your model, you'll need to re-optimize the "
    #                 "bandwidth.")
    # for i in tqdm(range(cfg.dataset.num_classes)):                        
    #     try:
    #         kdes[i] = KernelDensity(kernel='gaussian',
    #                             bandwidth=cfg.dataset.bandwidth) \
    #         .fit(X_train_features[class_inds[i]])
    #     except:
    #         #kdes[i] = KernelDensity(kernel='gaussian', bandwidth=cfg.dataset.bandwidth)
    #         # kdes[i] = kdes[i-1] #FIXME: Temp solution: Skip if class do not have any voice command
    #         kdes[i] = KernelDensity(kernel='gaussian',
    #                             bandwidth=cfg.dataset.bandwidth) \
    #         .fit(X_train_features)
    # # Get model predictions
    # print('Computing model predictions...')
    # all_probs, all_preds = predict_classes(net_target, nat_images, batch_size=batch_size)
    # preds_test_noisy = predict_classes(net_target, ben_images, batch_size=batch_size)
    # preds_test_adv = predict_classes(net_target, adv_images, batch_size=batch_size)

    # # Get density estimates
    # print('computing densities...')
    # densities_normal = score_samples(
    #     kdes,
    #     X_test_normal_features,
    #     preds_test_normal
    # )
    # densities_noisy = score_samples(
    #     kdes,
    #     X_test_noisy_features,
    #     preds_test_noisy
    # )
    # densities_adv = score_samples(
    #     kdes,
    #     X_test_adv_features,
    #     preds_test_adv
    # )

    # ## Z-score the uncertainty and density values
    # uncerts_normal_z, uncerts_adv_z, uncerts_noisy_z = normalize(
    #     uncerts_normal,
    #     uncerts_adv,
    #     uncerts_noisy
    # )
    # densities_normal_z, densities_adv_z, densities_noisy_z = normalize(
    #     densities_normal,
    #     densities_adv,
    #     densities_noisy
    # )

    # ## Build detector
    # values, labels, lr = train_lr(
    #     densities_pos=densities_adv_z,
    #     densities_neg=np.concatenate((densities_normal_z, densities_noisy_z)),
    #     uncerts_pos=uncerts_adv_z,
    #     uncerts_neg=np.concatenate((uncerts_normal_z, uncerts_noisy_z))
    # )
    # print(values.shape)

    # ## Evaluate detector
    # # Compute logistic regression model predictions
    # probs = lr.predict_proba(values)[:, 1]
    # # Compute AUC
    # n_samples = len(adv_images)
    # # The first 2/3 of 'probs' is the negative class (normal and noisy samples),
    # # and the last 1/3 is the positive class (adversarial samples).
    # _, _, auc_score = compute_roc(
    #     probs_neg=probs[:2 * n_samples],
    #     probs_pos=probs[2 * n_samples:]
    # )
    # # print(probs[:n_samples])
    # # print(probs[n_samples:2*n_samples])
    # # print(probs[2 * n_samples:])

    # print('1. Detector ROC-AUC score: %0.4f\n\n' % auc_score)


    # return probs[n_samples:]