# https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
# https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.fgsm


import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from helper import *
import json
from .attack_utils import *


def analyze_n_extract_feats(cfg, net, encoder, decoder, org_imgs, org_nois, img_conf_pre_feat, scale_factor, ntype, return_dist=False, repeat=10):
    rec_imgs = get_reconstructed_image(cfg, encoder, decoder, org_imgs)
    rec_nois = org_imgs - rec_imgs        
    img_conf_pre_feat[f"{ntype}_Org_Image"] = get_model_output(cfg, org_imgs, net, scale_factor)
    img_conf_pre_feat[f"{ntype}_Rec_Image"] = get_model_output(cfg, rec_imgs, net, scale_factor)
    img_conf_pre_feat[f"{ntype}_Org_Noise"] = get_model_output(cfg, org_nois, net, scale_factor)
    img_conf_pre_feat[f"{ntype}_Rec_Noise"] = get_model_output(cfg, rec_nois, net, scale_factor, return_dist, repeat)
    return img_conf_pre_feat



def analyze_n_extract_feats_unet(cfg, net, unet, org_imgs, org_nois, img_conf_pre_feat, scale_factor, ntype):
    rec_imgs = get_reconstructed_image_unet(cfg, unet, org_imgs)
    rec_nois = org_imgs - rec_imgs        
    img_conf_pre_feat[f"{ntype}_Org_Image"] = get_model_output(cfg, org_imgs, net, scale_factor)
    img_conf_pre_feat[f"{ntype}_Rec_Image"] = get_model_output(cfg, rec_imgs, net, scale_factor)
    img_conf_pre_feat[f"{ntype}_Org_Noise"] = get_model_output(cfg, org_nois, net, scale_factor)
    img_conf_pre_feat[f"{ntype}_Rec_Noise"] = get_model_output(cfg, rec_nois, net, scale_factor)
    return img_conf_pre_feat


def adv_attack(cfg, attack_type, net_dict, encoder, decoder, test_loader, env):
    
    # Set appropriate model
    print(f"Starting Attack Type: {attack_type}")
    # num_of_samples = cfg.n if attack_type != 'RN' else int(cfg.n*cfg.scale_rn)
    num_of_repeat = cfg.scale_rn if attack_type == 'RN' else 1
    num_of_samples = cfg.n * num_of_repeat
    print("Number of samples to generate: :", num_of_samples)

    label_data = []

    access_type, net_target, net_surrogate, scale_factor = configure_access(cfg, attack_type, net_dict)
    model_art = define_art_model(cfg, attack_type, net_surrogate)
    hook_handle = register_forward_hook(cfg, net_target, hook_fn)

    attack_ext = [f"{val}" for key, val in cfg.dataset.atk_cfg[attack_type].items()]
    attack_ext = "_".join(attack_ext)
    print(attack_ext)
    # Check in the temp data
    results_dict =  check_temp(cfg, access_type, attack_type, attack_ext, env, num_of_samples)
    if results_dict != None:
        return results_dict
    if attack_type == 'UAP':
        pre_gen_atk_data = handle_uap_attack(cfg, test_loader, model_art)

    device = torch.device(cfg.device)

    n=10
    correct_adv = 0
    correct_rec = 0
    iter_count = 0
    org_target_list = []
    magnet_score_list = []
    nadl_score_list = []
    combined_conf_pre_feat = []

    optimistic = True
    # Create a tqdm progress bar
    pbar = tqdm(total=num_of_samples)
    pbar.n = 0  # Reset progress to 0
    pbar.last_print_n = 0
    pbar.update(0)  # Refresh the display

    for idx, (org_images, target) in tqdm(enumerate(test_loader), ):
        for _ in range(num_of_repeat):
            if idx > 10*num_of_samples:
                optimistic = False
                print("Becoming Pessimistic!")
            
            target_class = target.max(1, keepdim=True)[1].detach().cpu().item()
            
            if cfg.target_classes_only and target_class not in cfg.dataset.target_classes:
                print("Skipping.... Not in the target class")
                continue

            # =====================================  Original Image  ================================================
            org_images, target = org_images.to(device), target.to(device)
            org_noises = org_images * 0.0
            #========================================================================================================


            # =====================================  Natural Image  ================================================
            nat_images, nat_noises = generate_attack(cfg, idx, 'GN', net_surrogate, org_images, target) #
            #========================================================================================================

            
            # =====================================  Adversarial Image  ================================================
            # if epsilon == 0.0:
            #     adv_images, adv_noises = generate_attack(cfg, idx, 'RN', net_surrogate, org_images, target) #
            if attack_type == 'C&W':
                adv_images, adv_noises = generate_attack(cfg, idx, attack_type, model_art, org_images, target)
            elif attack_type == 'UAP':
                adv_images, adv_noises = generate_attack(cfg, idx, attack_type, model_art, pre_gen_atk_data, target)
            else:
                adv_images, adv_noises = generate_attack(cfg, idx, attack_type, net_surrogate, org_images, target)
            if torch.max(adv_noises) == 0.0 and optimistic and attack_type != 'RN': 
                print("Attack failed, repeating!")
                continue
            
            #========================================================================================================

            # TODO : =====================================  Strong or Weak Attack  =========================================
            # if cfg.sync == 'weak' and epsilon > 0.0:
                # _, weak_noises = generate_attack(cfg, idx, 'RN', net_surrogate, org_images, target, eps=epsilon/cfg.weak_fraction)
                # # print("Before adding noise:", adv_images)
                # # print("Weak noises:", weak_noises)
                # adv_images = adv_images + weak_noises
                # # print("After adding noise:", adv_images)
                # adv_images = torch.clamp(adv_images, min=0, max=1)
                # # print("After clamping noise:", adv_images)
            # if cfg.sync == 'weak' and epsilon > 0.0:
            #     print("Weak Attack")
            #     adv_images = apply_real_world_transform(cfg, adv_images)
            #========================================================================================================


            # =====================================  Benign Image  ================================================
            flat_adv_noises = adv_noises.view(-1)
            shuffled_indices = torch.randperm(flat_adv_noises.size(0))
            ben_noises = flat_adv_noises[shuffled_indices].view(adv_noises.size())
            ben_images = org_images + ben_noises
            #========================================================================================================
            
            
            #============================ ===== Get Prediction & Features ===========================================
            img_conf_pre_feat = {}
            img_conf_pre_feat = analyze_n_extract_feats(cfg, net_target, encoder, decoder, org_images, org_noises, img_conf_pre_feat, scale_factor, ntype = 'Original')        
            img_conf_pre_feat = analyze_n_extract_feats(cfg, net_target, encoder, decoder, nat_images, nat_noises, img_conf_pre_feat, scale_factor, ntype = 'Natural', return_dist=True, repeat=100)
            img_conf_pre_feat = analyze_n_extract_feats(cfg, net_target, encoder, decoder, adv_images, adv_noises, img_conf_pre_feat, scale_factor, ntype = 'Adversarial', return_dist=True, repeat=100)
            img_conf_pre_feat = analyze_n_extract_feats(cfg, net_target, encoder, decoder, ben_images, ben_noises, img_conf_pre_feat, scale_factor, ntype = 'Benign', return_dist=True, repeat=100)
            #========================================================================================================

            
            if img_conf_pre_feat["Original_Org_Image"]['Pred'] != target_class:
                print("Only taking corrected predicted images")
                continue
            
            # Get the samples and target classes
            label_data.append(target_class)  

            # MagNet Detector
            magnet_score_list.append(magnet_attributes(img_conf_pre_feat))
            combined_conf_pre_feat.append(img_conf_pre_feat)
            

            org_adv_prediction = img_conf_pre_feat["Adversarial_Org_Image"]["Pred"]
            rec_adv_prediction = img_conf_pre_feat["Adversarial_Rec_Image"]["Pred"]
            org_target_list.append(target_class)

            # Accuracy checks
            if org_adv_prediction == target_class:
                correct_adv += 1
            if rec_adv_prediction == target_class:
                correct_rec += 1
            # Generated samples
            iter_count += 1
            if iter_count == num_of_samples:
                print(idx, iter_count)
                break
            pbar.update(1)
        if iter_count == num_of_samples:
            break

    pbar.close()
    # Remove the hook to avoid interference in future forward passes
    hook_handle.remove()
    final_acc_ben = round(correct_adv / float(num_of_samples) * 100, 3)
    print(f"Test Adversarial Accuracy = {correct_adv} / {num_of_samples} = {final_acc_ben}")
    final_acc_rec = round(correct_rec / float(num_of_samples) * 100, 3)
    print(f"Test Reconstruction Accuracy = {correct_rec} / {num_of_samples} = {final_acc_rec}")

    # Creating combined results
    all_conf_df = pd.DataFrame([])
    all_norm_df = pd.DataFrame([])
    all_feat_df = pd.DataFrame([])
    all_pred_df = pd.DataFrame([])

    # Getting the image data
    attack_data = {}
    for img_conf_pre_feat in combined_conf_pre_feat:
        for image_type, image_data in img_conf_pre_feat.items():
            attack_data.setdefault(image_type, []).append(image_data['Image'])
    attack_data = {k: np.array(v) for k, v in attack_data.items()}
    
    col_dict = {}
    for img_conf_pre_feat in combined_conf_pre_feat:
        
        ind_conf_data = []
        ind_conf_cols = []

        ind_feat_data = []
        ind_feat_cols = []
        
        ind_pred_cols = []
        ind_pred_data = []
        
        ind_norm_cols = []
        ind_norm_data = []

        for image_type, image_data in img_conf_pre_feat.items():
            col_dict[image_type] = {}
            
            norm_cols = [image_type+'_Norm']
            col_dict[image_type]['Norm'] = norm_cols
            ind_norm_cols += norm_cols
            ind_norm_data.append(image_data['Norm'])  
            
            pred_cols = [image_type+'_Pred']
            col_dict[image_type]['Pred'] = pred_cols
            ind_pred_cols += pred_cols
            ind_pred_data.append(image_data['Pred'])            

            conf_cols = [image_type+'_Conf_'+f'{i+1}' for i in range(image_data['Conf'].shape[0])]
            col_dict[image_type]['Conf'] = conf_cols
            ind_conf_cols += conf_cols
            ind_conf_data.append(image_data['Conf'])

            feat_cols = [image_type+'_Feat_'+f'{i+1}' for i in range(image_data['Feat'].shape[0])]
            col_dict[image_type]['Feat'] = feat_cols
            ind_feat_cols += feat_cols
            ind_feat_data.append(image_data['Feat'])        

        ind_pred_df = pd.Series(np.array(ind_pred_data), index=ind_pred_cols)
        ind_norm_df = pd.Series(np.array(ind_norm_data), index=ind_norm_cols)
        ind_conf_df = pd.Series(np.concatenate(ind_conf_data), index=ind_conf_cols)
        ind_feat_df = pd.Series(np.concatenate(ind_feat_data), index=ind_feat_cols)

        all_pred_df = pd.concat([all_pred_df, pd.DataFrame(ind_pred_df).T], axis=0, ignore_index=True)
        all_norm_df = pd.concat([all_norm_df, pd.DataFrame(ind_norm_df).T], axis=0, ignore_index=True)
        all_conf_df = pd.concat([all_conf_df, pd.DataFrame(ind_conf_df).T], axis=0, ignore_index=True)
        all_feat_df = pd.concat([all_feat_df, pd.DataFrame(ind_feat_df).T], axis=0, ignore_index=True)

    attack_analysis_df = pd.concat([all_pred_df, all_norm_df, all_conf_df, all_feat_df], axis=1)
    
    attack_analysis_df[['MagNet_Nat_L1', 
                        'MagNet_Ben_L1', 
                        'MagNet_Adv_L1', 
                        'MagNet_Nat_JSD', 
                        'MagNet_Ben_JSD', 
                        'MagNet_Adv_JSD']] = magnet_score_list

    attack_analysis_df['Target'] = org_target_list
    # attack_analysis_df['Epsilon'] = epsilon
    attack_analysis_df['Environment'] = env
    attack_analysis_df['Attack'] = attack_type
    attack_analysis_df['Access'] = access_type
    attack_analysis_df['Dataset'] = cfg.dataset.name
    attack_analysis_df['AeHidLay'] = cfg.dataset.ae_hidden_layers
    attack_analysis_df['AeLatDim'] = cfg.dataset.ae_latent_dim
    attack_analysis_df['AeNoiFact'] = cfg.dataset.noise_factor
    attack_analysis_df['EvalType'] = cfg.evaltype
    attack_analysis_df['Rep'] = cfg.rep
    attack_analysis_df['Scale'] = cfg.scale
    attack_analysis_df = attack_analysis_df.round(3)
   
    dict_dir = file_path(f"{cfg.results_dir}/data/col_dict.json")
    with open(dict_dir,"w") as f:
        json.dump(col_dict,f, indent=4)

    accuracy_dict = {"Attack" : attack_type, 
                     'Access' : access_type,
                    #  "Epsilon" : epsilon, 
                     "Environment" : env, 
                     "Accuracy": final_acc_ben,
                     "Dataset" : cfg.dataset.name}

                        
    results_dict = {
        'attack_analysis': attack_analysis_df,
        'attack_accuracy': accuracy_dict,
        'attack_data' : attack_data,
    }
    
    # attack_analysis_df.to_csv()
    ext = f"{cfg.dataset.name}_{access_type}_{attack_type}_{attack_ext}_{env}_{num_of_samples}"
    data_dir = file_path(f"{cfg.results_dir}/temp/attack_analysis_df_{ext}.csv")
    dict_dir = file_path(f"{cfg.results_dir}/temp/attack_accuracy_{ext}.json")
    attack_analysis_df.to_csv(data_dir, header=True, index=True)

    with open(dict_dir,"w") as f:
        json.dump(accuracy_dict,f, indent=4)

    if cfg.saveFig:
        plot_images_with_confidences_v2(cfg, access_type, attack_type, combined_conf_pre_feat, env, n)
    
    return results_dict