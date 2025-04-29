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

import numpy as np

@hydra.main(config_path="../config", config_name="config.yaml")
def detect_backdoor_examples(cfg: DictConfig) -> None:

    # Set up environment
    source_dir = Path(__file__).resolve().parent
    cfg = update_abs_path(cfg, source_dir)
    device = torch.device(cfg.device)

    dataset = cfg.dataset.name.upper()

    get_patch = cfg.get_patch #True
    patch_size = 95

    # Load autoencoder 
    encoder, decoder, pre_trained = get_autoencoder(cfg, pre_trained = True)
    assert pre_trained, "Model Not Trained"
    encoder.eval() 
    decoder.eval()
    
    pred_merged = pd.DataFrame([])
    scores_merged = pd.DataFrame([])
    roc_curve_merged = pd.DataFrame([])
    pr_curve_merged = pd.DataFrame([])
    tsne_data = {}
    n = 1000

    attack_analysis_df = pd.DataFrame([])
    
    for attack in cfg.backdoor_attacks:
        print("\n\n\n")
        print("*"*50)
        print(f"Backdoor Attacks: {attack}")
        print("*"*50)


        # Load Model
        print(f"Loading model from : ResNet-18_{dataset}_{attack}/")
        if cfg.dataset.name == 'cifar10':
            net = ResNet(18, num_feats=cfg.dataset.num_feats, num_classes=cfg.dataset.num_classes)
        elif cfg.dataset.name == 'mnist':
            net = BaselineMNISTNetwork(num_feats=cfg.dataset.num_feats, num_classes=cfg.dataset.num_classes)
        model_dir = f"{source_dir}/../artifacts/backdoorbox/models/{dataset}_{attack}/ckpt_epoch_200.pth"
        net = load_state_dict(net, model_dir)
        net.eval()
        hook_handle = register_forward_hook_backdoor(net, hook_fn)

        # Load datasets:
        print("\n\nLoading data from: ", source_dir)
        X_nat = np.load(f"{source_dir}/../artifacts/backdoorbox/data/{dataset}/backdoor_{dataset}_natural_{attack}_X.npy")[-n:]
        y_nat = np.load(f"{source_dir}/../artifacts/backdoorbox/data/{dataset}/backdoor_{dataset}_natural_{attack}_y.npy")[-n:]

        X_trg = np.load(f"{source_dir}/../artifacts/backdoorbox/data/{dataset}/backdoor_{dataset}_triggered_{attack}_X.npy")[0:n]
        y_trg = np.load(f"{source_dir}/../artifacts/backdoorbox/data/{dataset}/backdoor_{dataset}_triggered_{attack}_y.npy")[0:n]

        X_ben = np.load(f"{source_dir}/../artifacts/backdoorbox/data/{dataset}/backdoor_{dataset}_natural_{attack}_X.npy")[0:n]
        y_ben = np.load(f"{source_dir}/../artifacts/backdoorbox/data/{dataset}/backdoor_{dataset}_natural_{attack}_y.npy")[0:n]

        # # Set up evaluation dataset
        # X_trg = X_trg[0:250].copy()
        # X_ben = X_nat[0:250].copy()
        # X_nat = X_nat[250:]
        # y_trg = y_nat[0:250].copy()
        # y_nat = y_nat[250:]
        
        # Another version
        
        # actual_del_X_trg = (X_trg - X_nat).copy()
        # random_del_X_trg = shuffle_pixels(actual_del_X_trg)
        # X_ben = X_nat + random_del_X_trg


        # # Adding noise to the training data
        X_nat = add_noise_numpy(X_nat, noise_factor = 0.03) # 0.02) 
        # X_ben = add_noise_numpy(X_nat, noise_factor = 0.00)
        X_ben = add_noise_numpy(X_ben, noise_factor = 0.02)

        X_nat = torch.from_numpy(X_nat).float() #.double()
        X_trg = torch.from_numpy(X_trg).float() #.double()
        X_ben = torch.from_numpy(X_ben).float() #.double()
        print("===================================")

        print(X_nat.shape, X_trg.shape, X_ben.shape)

        # print("\n\nCheck the performance of the model:")
        # # Get feature representation using the model
        # _, y_pred_nat = get_feature_rep(cfg, X_nat, net, scale_factor=1.0)
        # _, y_pred_trg = get_feature_rep(cfg, X_trg, net, scale_factor=1.0)
        # _, y_pred_ben = get_feature_rep(cfg, X_ben, net, scale_factor=1.0)

        # print("Prediction on Actual Images:")
        # print(f"Train: {y_nat[0:20]}\n Nat label: {y_pred_nat[0:20]}")
        # print(f"Test:  {y_trg[0:20]}\nTrg label: {y_pred_trg[0:20]}\nBen label: {y_pred_ben[0:20]}")

        # # Calculate accuracy
        # accuracy_nat = np.mean(y_nat == y_pred_nat)*100
        # accuracy_trg = np.mean(y_ben == y_pred_trg)*100
        # accuracy_ben = np.mean(y_ben == y_pred_ben)*100

        # print("="*25)
        # print(f'Natural accuracy : {accuracy_nat}%')
        # print(f'Backdoor accuracy: {accuracy_trg}%')
        # print(f'Benign accuracy  : {accuracy_ben}%')
        # print("="*25)

        print("\n\nEvaluate detecction performance")
        # Get Reconstruction Error on each type of inputs
        if cfg.see_max:
            rec_noise_nat = torch.from_numpy(X_nat - X_nat)
            rec_noise_trg = torch.from_numpy(X_trg - X_nat)
            rec_noise_ben = torch.from_numpy(X_ben - X_nat)
        else:
            rec_noise_nat = get_recon_noise(cfg, X_nat, encoder, decoder, device, batch_size=32, return_org = get_patch, percentile=patch_size)
            rec_noise_trg = get_recon_noise(cfg, X_trg, encoder, decoder, device, batch_size=32, return_org = get_patch, percentile=patch_size)
            rec_noise_ben = get_recon_noise(cfg, X_ben, encoder, decoder, device, batch_size=32, return_org = get_patch, percentile=patch_size)


        # Get feature representation using the model
        indx = 0  # Index of the image to display
        noise_feat_nat, y_pred_nat = get_feature_rep(cfg, rec_noise_nat, net, scale_factor=1.0)
        nat_noise = np.transpose(rec_noise_nat[indx].detach().numpy(), (1, 2, 0))

        noise_feat_trg, y_pred_trg = get_feature_rep(cfg, rec_noise_trg, net, scale_factor=1.0)
        trg_noise = np.transpose(rec_noise_trg[indx].detach().numpy(), (1, 2, 0))
        
        noise_feat_ben, y_pred_ben = get_feature_rep(cfg, rec_noise_ben, net, scale_factor=1.0)
        ben_noise = np.transpose(rec_noise_ben[indx].detach().numpy(), (1, 2, 0))
        # print("Prediction on Recon Noise:")

        # Load noise features in a DataFrame
        Recon_NatNoise_Feats = [f'Natural_Rec_Noise_Feat_{i+1}' for i in range(cfg.dataset.num_feats)]
        Recon_BenNoise_Feats = [f'Benign_Rec_Noise_Feat_{i+1}' for i in range(cfg.dataset.num_feats)]
        Recon_AdvNoise_Feats = [f'Adversarial_Rec_Noise_Feat_{i+1}' for i in range(cfg.dataset.num_feats)]
        
        noise_feat_np = np.concatenate([noise_feat_nat, noise_feat_ben, noise_feat_trg], axis = 1)
        noise_feat_col = Recon_NatNoise_Feats + Recon_BenNoise_Feats + Recon_AdvNoise_Feats
        # print(noise_feat_nat.shape, noise_feat_np.shape)
        noise_feat_df = pd.DataFrame(noise_feat_np, columns = noise_feat_col)
        noise_feat_df["Attack"] = attack
        attack_analysis_df = pd.concat([attack_analysis_df, noise_feat_df], axis=0)
        print("attack_analysis_df.shape : ", attack_analysis_df.shape)


        # Get Baseline Detectors
        baseline_model_dict = get_all_anom_detect_models(cfg)
        baseline_model_dict = train_all_anom_detect_models(cfg, baseline_model_dict, noise_feat_nat, y_nat, attack)
        baseline_model_dict = load_all_anom_detect_models(cfg, attack)

    
        tsne_data[attack] = {"Original" : [X_ben[0:100], X_trg[0:100]],
                             "Feature" : [noise_feat_ben[0:100], noise_feat_trg[0:100]]}

        
        # Plot the figure..
        #------------------------------------------------
        # Transpose images from (C, H, W) to (H, W, C) for imshow
        nat_image = np.transpose(X_nat[indx], (1, 2, 0))
        trg_image = np.transpose(X_trg[indx], (1, 2, 0))
        ben_image = np.transpose(X_ben[indx], (1, 2, 0))

        # Correcting the subplot structure to fit all images
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))

        axes[0, 0].imshow(nat_image)
        axes[0, 0].set_title('Natural Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(trg_image)
        axes[0, 1].set_title('Triggered Image')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(ben_image)
        axes[0, 2].set_title('Benign Image')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(nat_image - nat_noise)
        axes[1, 0].set_title('Natural Noise')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(trg_image - trg_noise)
        axes[1, 1].set_title('Triggered Noise')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(ben_image - ben_noise)
        axes[1, 2].set_title('Benign Noise')
        axes[1, 2].axis('off')


        axes[2, 0].imshow(nat_noise)
        axes[2, 0].set_title('Natural Noise')
        axes[2, 0].axis('off')

        axes[2, 1].imshow(trg_noise)
        axes[2, 1].set_title('Target Noise')
        axes[2, 1].axis('off')

        axes[2, 2].imshow(ben_noise)
        axes[2, 2].set_title('Benign Noise')
        axes[2, 2].axis('off')

        plt.tight_layout()
        save_dir = Path(f"{source_dir}/../artifacts/backdoorbox/plots/Backdoor_{dataset}_{attack}_{get_patch}.jpg")
        save_dir.parent.mkdir(parents=True,exist_ok=True)
        plt.savefig(save_dir)
        plt.show()
        print(f"\n\nImage saved at: {source_dir}/Backdoor_{dataset}_{attack}.jpg")
        #------------------------------------------------
        # Test Baseline Models
        X_test = np.concatenate((noise_feat_ben, noise_feat_trg), axis=0)
        y_cls = att_analysis_adv_conf['Adversarial_Rec_Image_Pred'].values #FIXME
        #--------------------------------
        y_prediction_dict = {}
        for detector_list in [baseline_model_dict.items()]: 
            for det_model_pair in detector_list:
                detector, model = det_model_pair if isinstance(det_model_pair, tuple) else (det_model_pair, None)
                y_prediction_dict[detector] = get_prediction(cfg, model, X_test, y_cls, detector)
                if cfg.magsec:
                    y_prediction_dict[detector] = (y_prediction_dict[detector] + y_magnet_scaled)/2
        #--------------------------------



        y_test_ben = np.zeros(noise_feat_ben.shape[0])
        y_test_trg = np.ones(noise_feat_trg.shape[0])
        y_test = np.concatenate((y_test_ben, y_test_trg), axis=0)
        
        #Dummy var
        X_feat = None
        eps=1.0
        env='clean'
        if cfg.verbose:
            print(f"Shape of X_test_ben: {noise_feat_ben.shape}")
            print(f"Shape of X_test_adv: {noise_feat_trg.shape}")
            print(f"Shape of X_test: {X_test.shape}")
            print(f"Shape of y_test: {y_test.shape}")

        for detector_list in [baseline_model_dict.items()]: #, cfg.models.tsfel_feats]:
            for det_model_pair in detector_list:
                detector, model = det_model_pair if isinstance(det_model_pair, tuple) else (det_model_pair, None)

                # #-------------------
                # y_pred = y_prediction_dict[detector]
                # #-------------------
                predictions_df, roc_curve_df, pr_curve_df, auc_scores_df = test_baseline(cfg, model, X_feat, y_pred, y_test, y_trg, eps, env, attack, detector)
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

    # Save results and pred data
    scores_dir = file_path(f"{cfg.results_dir}/detection/final_scores_backdoor_{cfg.blackbox}_{cfg.evaltype}_{cfg.evalenv}_{cfg.p_step}.csv")
    scores_merged.to_csv(scores_dir, header=True, index=True)

    pred_dir = file_path(f"{cfg.results_dir}/detection/final_predictions_backdoor_{cfg.blackbox}_{cfg.evaltype}_{cfg.evalenv}_{cfg.p_step}.csv")
    pred_merged.to_csv(pred_dir, header=True, index=True)

    pred_dir = file_path(f"{cfg.results_dir}/detection/final_roc_curves_backdoor_{cfg.blackbox}_{cfg.evaltype}_{cfg.evalenv}_{cfg.p_step}.csv")
    roc_curve_merged.to_csv(pred_dir, header=True, index=True)

    pred_dir = file_path(f"{cfg.results_dir}/detection/final_pr_curves_backdoor_{cfg.blackbox}_{cfg.evaltype}_{cfg.evalenv}_{cfg.p_step}.csv")
    pr_curve_merged.to_csv(pred_dir, header=True, index=True)

    print("Evaluation Complete", pred_dir)
    plot_tsne(cfg, tsne_data, attack_type='backdoor')
    
    data_dir = file_path(f"{cfg.results_dir}/data/attack_analysis_df_backdoor_{dataset.lower()}_{cfg.blackbox}.csv")
    attack_analysis_df.to_csv(data_dir, header=True, index=True)
# Main function
if __name__ == '__main__':
    detect_backdoor_examples()

# python detect_backdoor_examples.py
