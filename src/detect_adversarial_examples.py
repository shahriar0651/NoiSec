import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


from datasets import *
from models import *
from attacks import *
from helper import *
import json


@hydra.main(config_path="../config", config_name="config.yaml")
def detect_adv_examples(cfg: DictConfig) -> None:

    # Set up environment
    source_dir = Path(__file__).resolve().parent
    cfg = update_abs_path(cfg, source_dir)
    device = torch.device(cfg.device)

    dataset = cfg.dataset.name
    num_feats = cfg.dataset.num_feats

    pred_merged = pd.DataFrame([])
    scores_merged = pd.DataFrame([])
    roc_curve_merged = pd.DataFrame([])
    pr_curve_merged = pd.DataFrame([])
    
    # For Artifacts------------------------
    cfg.dataset.batch_size = 1
    # Train the anomaly detector on test dataset
    X_train_img, Y_train_img = get_random_samples(cfg, data_loader, data_type='test', n=cfg.aux_samples) 
    tsne_data = {}  
    #--------------------------------------

    # Get the attack data generated in the previous step
    with open(f"{cfg.results_dir}/data/col_dict.json", 'r') as file:
        col_dict = json.load(file)
    data_dir = Path(f"{cfg.results_dir}/data/attack_analysis_df_{dataset}_{cfg.blackbox}.csv")
    att_analysis_all_env = pd.read_csv(data_dir, index_col=0)
    print("Unique Attacks: ", att_analysis_all_env['Attack'].value_counts())

    att_analysis_evalype_all_env = filter_dataset_by_evaltype(cfg, att_analysis_all_env, col_dict)
    print("Unique Attacks: ", att_analysis_evalype_all_env['Attack'].value_counts())

    if cfg.verbose:
        print(f"Starting the evaluation with evaltype: {cfg.evaltype}")
        print(f"Shape of att_analysis_all_env : {att_analysis_all_env.shape}")
        print(f"Shape of att_analysis_evalype_all_env : {att_analysis_evalype_all_env.shape}")
    
    # Select the features of interest
    org_or_rec = 'Org' if cfg.see_max else 'Rec' # Org --> Ideal, Rec --> Practical
    nat_noi_feats = col_dict[f'Natural_{org_or_rec}_Noise'][cfg.rep]
    ben_noi_feats = col_dict[f'Benign_{org_or_rec}_Noise'][cfg.rep]
    adv_noi_feats = col_dict[f'Adversarial_{org_or_rec}_Noise'][cfg.rep]
    adv_img_conf = col_dict['Adversarial_Org_Image']["Conf"]
    ben_img_conf = col_dict['Benign_Org_Image']["Conf"]

    for n_components in list(map(int, np.linspace(1, cfg.dataset.num_classes, cfg.n_comp_step))): #FIXME: Remove the loop
        print("n_components: ", n_components)
        # Run model training for baseline models
        for env in cfg.environments:
            print(f"Environment: {env}")
            att_analysis = att_analysis_all_env[att_analysis_all_env['Environment'] == env]
            att_analysis_eval = att_analysis_evalype_all_env[att_analysis_evalype_all_env['Environment'] == env]
            if cfg.verbose:
                print(f"Shape of att_analysis : {att_analysis.shape}")
                print(f"Shape of att_analysis_eval : {att_analysis_eval.shape}")
            
           
            # Training the baseline models---------------------------------------------------------------
            if cfg.basetype in ['train', 'both']:
                # Load training data for the anomaly detector
                filter_eps = att_analysis_eval['Attack'] == 'RN'
                print("Total Count : ", filter_eps.astype(int).sum())
                X_train  = att_analysis_eval.where(filter_eps)[adv_noi_feats].dropna().values
                Y_train  = att_analysis_eval.where(filter_eps)["Target"].dropna().values

                if cfg.verbose:
                    print(f"X_train--> Mean: {np.mean(X_train)}, Std: {np.std(X_train)}")
                    print("X_train, Y_train : ", X_train.shape, Y_train.shape)

                baseline_model_dict = get_all_anom_detect_models(cfg, n_components)
                baseline_model_dict = train_all_anom_detect_models(cfg, baseline_model_dict, X_train, Y_train, env)
            
            # Testing the baseline models---------------------------------------------------------------            
            if cfg.basetype in ['test', 'both']:
                baseline_model_dict = load_all_anom_detect_models(cfg, env)
                for attack in att_analysis_eval['Attack'].unique():
                    print(f"\n\nAttack: {attack}")
                    if attack == 'RN':
                        continue
                    # Benign samples
                    if attack == 'BadNet': 
                        if cfg.blackbox:
                            continue
                        net_target, pre_trained = get_conv_net(cfg, model_type='badnet', pre_trained=True) 
                        scale_factor = 255.0
                    else:
                        net_target, pre_trained =  get_conv_net(cfg, model_type='target', pre_trained=True)
                        scale_factor = 1.0
                    net_target.eval()
                    assert pre_trained, "Model Not Trained"

                    print("Starting Manda")
                    selected_indices = get_index_by_evaltype(cfg, att_analysis_all_env, attack)
                    y_manda = get_manda_score(cfg, net_target, X_train_img, Y_train_img, selected_indices, attack, env, file_path, scale_factor)
                    y_artifacts = get_artifacts_score(cfg, net_target, X_train_img, Y_train_img, selected_indices, attack, env, file_path, scale_factor)

                    print("att_analysis_eval: ", att_analysis_eval.shape)
                    print(att_analysis_eval['Attack'].value_counts(), attack)
                    filter_att = att_analysis_eval['Attack'] == attack
                    att_analysis_attack = att_analysis_eval.where(filter_att).dropna()
                    print("att_analysis_attack: ", att_analysis_attack.shape)

                    #--- adding for low conf sample ---
                    max_ben_prb = att_analysis_attack[ben_img_conf].max(axis=1)
                    # print("max_ben_prb : ", max_ben_prb)
                    min_of_max_ben_prb = np.min(max_ben_prb)
                    max_ben_prb = max_ben_prb - min_of_max_ben_prb
                    max_of_max_ben_prb = np.max(max_ben_prb)
                    max_ben_prb/=max_of_max_ben_prb #FIXME :Scaling max to 1
                    prob_list =  np.arange(0, 101, cfg.p_step)/100

                    print("ben_prob_list: ", prob_list)
                    for index_ben in range(len(prob_list)-1):
                        p_min_ben = prob_list[index_ben]
                        p_max_ben = prob_list[index_ben+1]
                        filter_min = max_ben_prb > p_min_ben
                        filter_max = max_ben_prb <= p_max_ben
                        att_analysis_ben_conf = att_analysis_attack
                        print("att_analysis_ben_conf: ", att_analysis_ben_conf.shape)
                        # ----------------
                        
                        if cfg.evalenv == 'NatNoise': # Evaluate against "No Noise"
                            X_test_ben = att_analysis_ben_conf[nat_noi_feats].values
                            y_magnet_ben = att_analysis_ben_conf[['MagNet_Nat_L1', 'MagNet_Nat_JSD']].values

                        elif cfg.evalenv == 'BenPerturb': # Evaluate against "Benign Noise"
                            X_test_ben = att_analysis_ben_conf[ben_noi_feats].values
                            y_magnet_ben = att_analysis_ben_conf[['MagNet_Ben_L1', 'MagNet_Ben_JSD']].values
                        
                        max_adv_prb = att_analysis_ben_conf[adv_img_conf].max(axis=1)                    
                        min_of_max_adv_prb = np.min(max_adv_prb)
                        max_adv_prb = max_adv_prb-min_of_max_adv_prb
                        max_of_max_adv_prb = np.max(max_adv_prb)
                        max_adv_prb/=max_of_max_adv_prb #FIXME :Scaling max to 1

                        print("max of np.max(max_adv_prb): ", np.max(max_adv_prb))
                        print("min of np.max(max_adv_prb): ", np.min(max_adv_prb))

                        for index_adv in range(len(prob_list)-1):
                            p_min_adv = prob_list[index_adv]
                            p_max_adv = prob_list[index_adv+1]
                            print(f"\n P_mim/max (adv): {p_min_adv}, {p_max_adv}")
                            filter_min = max_adv_prb >= p_min_adv
                            filter_max = max_adv_prb <= p_max_adv
                            att_analysis_adv_conf = att_analysis_ben_conf
                            print("att_analysis_ben_conf: ", att_analysis_ben_conf.shape)
                            print("att_analysis_adv_conf: ", att_analysis_adv_conf.shape)
                            X_test_adv = att_analysis_adv_conf[adv_noi_feats].values
                            y_cls = att_analysis_adv_conf['Adversarial_Rec_Image_Pred'].values #FIXME

                            #--------------- Vis TSNE -----------------#
                            # attack_data_Natural_Org_Image_
                            data_dir = file_path(f"{cfg.results_dir}/data/{attack}/attack_data_Benign_Rec_Noise_{env}_{cfg.blackbox}.npy")
                            ben_images = np.load(data_dir)
                            data_dir = file_path(f"{cfg.results_dir}/data/{attack}/attack_data_Adversarial_Rec_Noise_{env}_{cfg.blackbox}.npy")
                            adv_images = np.load(data_dir)

                            tsne_data[attack] = {"Original" : [ben_images, adv_images],
                                                    "Feature" : [X_test_ben, X_test_ben]}
                            
                            get_tsne_vis(cfg, attack, ben_images, adv_images, label_1 = "Benign", label_2 = "Malicious", dim = "original")
                            get_tsne_vis(cfg, attack, X_test_ben, X_test_adv, label_1 = "Benign", label_2 = "Malicious", dim = "feature")
                            #-------------------------------------------# 

                            X_test = np.concatenate((X_test_ben, X_test_adv), axis=0)

                            y_magnet_adv = att_analysis_adv_conf[['MagNet_Adv_L1', 'MagNet_Adv_JSD']].values
                            y_magnet = np.concatenate((y_magnet_ben, y_magnet_adv), axis=0)
                            y_test_ben = np.zeros(X_test_ben.shape[0])
                            y_test_adv = np.ones(X_test_adv.shape[0])
                            y_test = np.concatenate((y_test_ben, y_test_adv), axis=0)
                            

                            # Add MagNet with NoiSec---------
                            y_magnet_scaled = MinMaxScaler().fit(y_magnet_ben).transform(y_magnet)[:,1]
                            print("y_magnet_scaled: ", y_magnet_scaled.shape)

                            #--------------------------------
                            y_prediction_dict = {}
                            for detector_list in [baseline_model_dict.items()]: 
                                for det_model_pair in detector_list:
                                    detector, model = det_model_pair if isinstance(det_model_pair, tuple) else (det_model_pair, None)
                                    y_prediction_dict[detector] = get_prediction(cfg, model, X_test, y_cls, detector)
                                    if cfg.magsec:
                                        y_prediction_dict[detector] = (y_prediction_dict[detector].flatten() + y_magnet_scaled.flatten())/2
                            #--------------------------------

                            X_feat = None
                            if cfg.verbose:
                                print(f"Shape of X_test_ben: {X_test_ben.shape}")
                                print(f"Shape of X_test_adv: {X_test_adv.shape}")
                                print(f"Shape of X_test: {X_test.shape}")
                                print(f"Shape of y_test: {y_test.shape}")
                                print(f"Shape of y_artifacts: {y_artifacts.shape}")

                            if X_test_ben.shape[0] == 0 or X_test_adv.shape[0] == 0:
                                print("Limited data to run the testing!")
                                continue
                    
                            for indx, detector in enumerate(cfg.models.manda):
                                print("Detection: ", detector)
                                predictions_df, roc_curve_df, pr_curve_df, auc_scores_df = test_baseline(cfg, None, None, y_manda, y_test, y_cls, env, attack, detector, p_max_ben, p_max_adv)
                                pred_merged = pd.concat([pred_merged, predictions_df], axis=0, ignore_index=True)
                                roc_curve_merged = pd.concat([roc_curve_merged, roc_curve_df], axis=0, ignore_index=True)
                                pr_curve_merged = pd.concat([pr_curve_merged, pr_curve_df], axis=0, ignore_index=True)
                                scores_merged = pd.concat([scores_merged, auc_scores_df], axis=1, ignore_index=True)

                            for indx, detector in enumerate(cfg.models.artifacts):
                                print("Detection: ", detector)
                                predictions_df, roc_curve_df, pr_curve_df, auc_scores_df = test_baseline(cfg, None, None, y_artifacts, y_test, y_cls, env, attack, detector, p_max_ben, p_max_adv)
                                pred_merged = pd.concat([pred_merged, predictions_df], axis=0, ignore_index=True)
                                roc_curve_merged = pd.concat([roc_curve_merged, roc_curve_df], axis=0, ignore_index=True)
                                pr_curve_merged = pd.concat([pr_curve_merged, pr_curve_df], axis=0, ignore_index=True)
                                scores_merged = pd.concat([scores_merged, auc_scores_df], axis=1, ignore_index=True)

                            for indx, detector in enumerate(cfg.models.magnet_det):
                                print("Detection: ", detector)
                                predictions_df, roc_curve_df, pr_curve_df, auc_scores_df = test_baseline(cfg, None, None, y_magnet[:,indx], y_test, y_cls, env, attack, detector, p_max_ben, p_max_adv)
                                pred_merged = pd.concat([pred_merged, predictions_df], axis=0, ignore_index=True)
                                roc_curve_merged = pd.concat([roc_curve_merged, roc_curve_df], axis=0, ignore_index=True)
                                pr_curve_merged = pd.concat([pr_curve_merged, pr_curve_df], axis=0, ignore_index=True)
                                scores_merged = pd.concat([scores_merged, auc_scores_df], axis=1, ignore_index=True)
                
                            for detector_list in [baseline_model_dict.items()]: #, cfg.models.tsfel_feats]:
                                for det_model_pair in detector_list:
                                    detector, model = det_model_pair if isinstance(det_model_pair, tuple) else (det_model_pair, None)
                                    y_pred = y_prediction_dict[detector]
                                    print("Detection: ", detector, y_pred.shape)
                                    predictions_df, roc_curve_df, pr_curve_df, auc_scores_df = test_baseline(cfg, model, X_feat, y_pred, y_test, y_cls, env, attack, detector, p_max_ben, p_max_adv)
                                    pred_merged = pd.concat([pred_merged, predictions_df], axis=0, ignore_index=True)
                                    roc_curve_merged = pd.concat([roc_curve_merged, roc_curve_df], axis=0, ignore_index=True)
                                    pr_curve_merged = pd.concat([pr_curve_merged, pr_curve_df], axis=0, ignore_index=True)
                                    scores_merged = pd.concat([scores_merged, auc_scores_df], axis=1, ignore_index=True)
                                    print("scores_merged.shape : ", scores_merged.shape)
                            
                           
        scores_merged = scores_merged.T

        scores_merged["EvalType"] = cfg.evaltype
        pred_merged["EvalType"] = cfg.evaltype
        roc_curve_merged["EvalType"] = cfg.evaltype
        pr_curve_merged["EvalType"] = cfg.evaltype

        scores_merged["Scale"] = cfg.scale
        pred_merged["Scale"] = cfg.scale
        roc_curve_merged["Scale"] = cfg.scale
        pr_curve_merged["Scale"] = cfg.scale

        scores_merged["Rep"] = cfg.rep
        pred_merged["Rep"] = cfg.rep
        roc_curve_merged["Rep"] = cfg.rep
        pr_curve_merged["Rep"] = cfg.rep

        scores_merged["AeHidLay"] = cfg.dataset.ae_hidden_layers
        pred_merged["AeHidLay"] = cfg.dataset.ae_hidden_layers
        roc_curve_merged["AeHidLay"] = cfg.dataset.ae_hidden_layers
        pr_curve_merged["AeHidLay"] = cfg.dataset.ae_hidden_layers

        scores_merged["AeLatDim"] = cfg.dataset.ae_latent_dim
        pred_merged["AeLatDim"] = cfg.dataset.ae_latent_dim
        roc_curve_merged["AeLatDim"] = cfg.dataset.ae_latent_dim
        pr_curve_merged["AeLatDim"] = cfg.dataset.ae_latent_dim

        scores_merged["AeNoiFact"] = cfg.dataset.noise_factor
        pred_merged["AeNoiFact"] = cfg.dataset.noise_factor
        roc_curve_merged["AeNoiFact"] = cfg.dataset.noise_factor
        pr_curve_merged["AeNoiFact"] = cfg.dataset.noise_factor

        if cfg.adaptive:
            scores_merged['AdaptiveAttack'] = cfg.adaptive_attack
            scores_merged['AdaptiveEps'] = cfg.adaptive_eps
        
        # Save results and pred data
        scores_dir = file_path(f"{cfg.results_dir}/detection/final_scores_adversarial_{cfg.blackbox}_{cfg.evaltype}_{cfg.evalenv}_{cfg.p_step}_{n_components}.csv")
        scores_merged.to_csv(scores_dir, header=True, index=True)

        print("Unique Attacks: ", scores_merged['Attack'].value_counts())

        pred_dir = file_path(f"{cfg.results_dir}/detection/final_predictions_adversarial_{cfg.blackbox}_{cfg.evaltype}_{cfg.evalenv}_{cfg.p_step}_{n_components}.csv")
        pred_merged.to_csv(pred_dir, header=True, index=True)

        pred_dir = file_path(f"{cfg.results_dir}/detection/final_roc_curves_adversarial_{cfg.blackbox}_{cfg.evaltype}_{cfg.evalenv}_{cfg.p_step}_{n_components}.csv")
        roc_curve_merged.to_csv(pred_dir, header=True, index=True)

        pred_dir = file_path(f"{cfg.results_dir}/detection/final_pr_curves_adversarial_{cfg.blackbox}_{cfg.evaltype}_{cfg.evalenv}_{cfg.p_step}_{n_components}.csv")
        pr_curve_merged.to_csv(pred_dir, header=True, index=True)

    print(f"Evaluation Complete", pred_dir)
    plot_tsne(cfg, tsne_data)

# Main function
if __name__ == '__main__':
    detect_adv_examples()