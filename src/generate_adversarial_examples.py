import os
from pathlib import Path
import hydra
import torch
from omegaconf import DictConfig
import numpy as np
import seaborn as sns
from itertools import product
import copy
from datasets import *
from models import *
from attacks import *
from helper import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


@hydra.main(config_path="../config", config_name="config.yaml")
def generate_adv_examples(cfg: DictConfig) -> None:
    

    # Set up environment
    source_dir = Path(__file__).resolve().parent
    cfg = update_abs_path(cfg, source_dir)
    torch.manual_seed(42)

    # Run the pipeline
    cfg.dataset.batch_size = 1
    dataset = cfg.dataset.name
    #---------------------------------------------------------#
    #               Load dataset
    #---------------------------------------------------------#
    train_loader, test_loader, _, _ = data_loader(cfg, model_type='target')
    X_train_org, Y_train_org = get_X_Y(test_loader) #get_X_Y(val_loader)
    X_train_org = X_train_org[:cfg.n]
    Y_train_org = Y_train_org[:cfg.n]


    #---------------------------------------------------------#
    #               Load model
    #---------------------------------------------------------#
    net_dict = {}
    model_type_list = ['target', 'surrogate']
    if 'BadNet' in cfg.adv_attacks:
        model_type_list.append('badnet') #Only add badnet model if needed
    for model_type in model_type_list:
        net, pre_trained = get_conv_net(cfg, model_type=model_type, pre_trained=True) # FIXME : Train Classifier
        if pre_trained:
            net_dict[model_type] = net 
            net_dict[model_type].eval()
        else:
            print("Model Not Trained")
            return None 
        
    encoder, decoder, pre_trained = get_autoencoder(cfg, pre_trained = True)
    assert pre_trained, "Train the autoencoder first!"
    encoder.eval() 
    decoder.eval()    



    #---------------------------------------------------------#
    #               Generate attacks
    #---------------------------------------------------------#
    # Run test for each epsilon
    print("Generating attacks ..... ")
    accuracies = []
    attack_analysis_df = pd.DataFrame([])
    for env, attack in product(cfg.environments, cfg.adv_attacks):
        
        print(f"Starting with attack: {attack}")
        print("\nStarting with {} environment, {} attack".format(env, attack))

        #---------------------- genereate adversarial examples ----------------------#
        results_dict = adv_attack(cfg, attack, net_dict, encoder, decoder, test_loader, env)               
        #----------------------------------------------------------------------------#

        #---------------------- Save adversarial examples --------------------------#
        for key, val in results_dict.items():
            if key == 'attack_analysis':
                attack_analysis_df = pd.concat([attack_analysis_df, val], axis=0)
            elif key == 'attack_accuracy':
                accuracies.append(val)
            elif key == 'attack_data':
                for data_type, data in val.items():
                    data_dir = file_path(f"{cfg.results_dir}/data/{attack}/{key}_{data_type}_{env}_{cfg.blackbox}.npy")
                    np.save(data_dir, data)
                    print(f"\n\nSaved data in {data_dir}: {data.shape}")
            else:
                data_dir = file_path(f"{cfg.results_dir}/data/{attack}/{key}_{env}_{cfg.blackbox}.npy")
                np.save(data_dir, val)
    
    #---------------------------------------------------------#
    #               Save data
    #---------------------------------------------------------#
    if cfg.adaptive:
        attack_analysis_df['AdaptiveAttack'] = cfg.adaptive_attack
        attack_analysis_df['AdaptiveEps'] = cfg.adaptive_eps
    data_dir = file_path(f"{cfg.results_dir}/data/attack_analysis_df_{dataset}_{cfg.blackbox}.csv")
    attack_analysis_df.to_csv(data_dir, header=True, index=True)
    
    attack_accuracy_df = pd.DataFrame(accuracies)
    data_dir = file_path(f"{cfg.results_dir}/data/attack_accuracy_df_{dataset}_{cfg.blackbox}.csv")
    attack_accuracy_df.to_csv(data_dir, header=True, index=True)

    # Plot the acc vs eps graph
    for env in attack_accuracy_df['Environment'].unique():
        acc = attack_accuracy_df[attack_accuracy_df['Environment'] == env]
        print(acc)
        # plt.figure(figsize=(5,5))
        # sns.lineplot(acc, x = 'Epsilon', y = 'Accuracy', style = 'Attack',  hue = 'Attack', markers=True)
        # plt.title(f"Accuracy under different attacks with blackbox = {cfg.blackbox} - ({dataset}))")
        # plt.xlabel("Epsilon / Equivalent Factor")
        # plt.ylabel("Accuracy")
        # plt.tight_layout()
        # fig_dir = file_path(f"{cfg.results_dir}/plots/acc_vs_eps_{dataset}_{env}_{cfg.blackbox}.jpg")
        # plt.savefig(fig_dir, dpi = 350)
        # plt.show()

    print(accuracies)

# Main function
if __name__ == '__main__':
    generate_adv_examples()

