
#%%
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import time
from sklearn.metrics import accuracy_score
import matplotlib.cm as cm


from datasets import *
from models import *
from attacks import *
from helper import *


@hydra.main(config_path="../config", config_name="config.yaml")
def anayze_results(cfg: DictConfig) -> None:

    # Set up environment
    # cfg.autoencoder_type = 'UNet' 
    source_dir = Path(__file__).resolve().parent
    cfg = update_abs_path(cfg, source_dir)

    dataset = cfg.dataset.name
    # epsilons = cfg.dataset.epsilons
    # classes = cfg.dataset.classes

    attackType = cfg.attackType
    markers = cfg.markers
    evaltype = cfg.evaltype
    evalenv = cfg.evalenv

    dataset_name_dict = {chestmnist : chestmnist,
                     'fashion' : 'F-MNIST',
                     'cifar10' : 'CIFAR-10',
                     'gtsrb' : 'GTSRB',
                     'speech' : 'SPEECHCOMMAND',
                     'cancer' : 'CANCER',
                     'radiomlv1' : 'RadioMLv1',
                     'radiomlv2' : 'RadioMLv2',
                     'radiomlv3' : 'RadioMLv3',
                     'robofi' : 'RoboFi',
                     'activity' : 'Activity',
                     'chestmnist' : 'ChestMNIST',
                     }
    detector_name_dict ={
        'Artifacts' : 'Artifact',
                         'MagNet_L1' : 'MagNet(L1)',
                         'MagNet_JSD' : 'MagNet(JSD)',
                         'KNN' : 'NoiSec(KNN2)',
                         'KNN_L1' : 'NoiSec(KNN1)',
                         'KNN_Cos' : 'NoiSec(KNNC)',
                         'GMM' : 'NoiSec(GMM)',
                         'iKNN' : 'NoiSec(iKNN)',
                         'iGMM' : 'NoiSec(iGMM)',
                         'Max' : 'NoiSec(MAX)',
                         'STD' : 'NoiSec(STD)',
                         'NADL' : 'NoiSec(NADL)',
                         'Manda' : 'Manda',
                         'NADL' : 'NADL',
                         }
    replacement_dict = {   
        'Manda' : 'Manda',     
        'Artifacts' : 'Artifact',
        'KNN': 'NoiSec\n(KNN-L2)',
        'KNN_L1': 'NoiSec\n(KNN-L1)',
        'GMM': 'NoiSec\n(GMM)',
        'iKNN': 'NoiSec\n(iKNN)',
        'iGMM': 'NoiSec\n(iGMM)',
        'Max': 'NoiSec\n(MAX)',
        'STD': 'NoiSec\n(STD)',
        'NADL': 'NoiSec\n(xAD)',
        'MagNet_L1': 'MagNet\n(L1)',
        'MagNet_JSD': 'MagNet\n(JSD)'
    }
    # markers = ['o', 's', '^', 'x', '+']

    with open(f"{cfg.results_dir}/data/col_dict.json", 'r') as file:
        col_dict = json.load(file)
    
    adv_img_preds_col = col_dict['Adversarial_Org_Image']['Pred']
    adv_noi_feat_col = col_dict['Adversarial_Rec_Noise'][cfg.rep]
    adv_per_norm_col = col_dict['Adversarial_Org_Noise']['Norm']
    org_targets_col = ['Target']

    print("List of n_components : ", list(map(int, np.linspace(1, cfg.dataset.num_classes, cfg.n_comp_step))))
    for n_components in list(map(int, np.linspace(1, cfg.dataset.num_classes, cfg.n_comp_step))):
 
        for env in cfg.environments:
            print(f"Environment: {env}")


            # Results of the attack generation
            attack_analysis_df = pd.read_csv(f"{cfg.results_dir}/data/attack_analysis_df_{dataset}_{cfg.blackbox}.csv", index_col=0)
            print(f"results_gen_df.shape: {attack_analysis_df.shape}", attack_analysis_df)
            xbox = 'Black-box' if cfg.blackbox else 'White-box'
            plot_dir = Path(f'{cfg.results_dir}/plots/Overall_{n_components}/{xbox}')
            plot_dir.mkdir(parents=True, exist_ok=True)
            print("plot_dir: ", plot_dir)

            roc_dir = plot_dir / 'roc_curves'
            roc_dir.mkdir(parents=True, exist_ok=True)
            # results_gen_df = results_gen_df[results_gen_df['Attack'] != 'GN']
            results_gen_env_df = attack_analysis_df.copy()
            results_gen_df = results_gen_env_df[results_gen_env_df['Environment'] == env]
            print(f"results_gen_df.shape: {results_gen_df.shape}")

            # Create table for L-2 norm and Accuracy
            gen_data = pd.DataFrame()
            for dataset in results_gen_df['Dataset'].unique():
                for attack in results_gen_df['Attack'].unique():
                    print("attack : ", attack)
                    # for eps in results_gen_df['Epsilon'].unique():
                    filter_0 = results_gen_df['Dataset'] == dataset
                    filter_1 = results_gen_df['Attack'] == attack
                    # filter_2 = results_gen_df['Epsilon'] == eps
                    df = results_gen_df.where(filter_0 & filter_1)[adv_img_preds_col+org_targets_col+adv_per_norm_col]
                    df = df.dropna()
                    y_pred = df[adv_img_preds_col].astype(int)
                    y_true = df[org_targets_col].astype(int)
                    accuracy = accuracy_score(y_true, y_pred)
                    # try:
                    #     accuracy = accuracy_score(y_true, y_pred)
                    # except IndexError:
                    #     accuracy = 0
                    gen_data.loc[attack, f"L2"] = float(round(df[adv_per_norm_col].mean(), 2))
                    gen_data.loc[attack, f"Acc"] = float(round(accuracy*100, 3))
            gen_data.insert(0, 'Attack', gen_data.index)
            # gen_data["ASR"] = 100 - gen_data["Acc"]
            gen_data.to_csv(f"{cfg.results_dir}/data/attack_gen_acc_norm_{xbox}_{env}.csv")
            print(gen_data)
    
            #%%
            # # Plot the accuracy and L2 norm
            # fig, axes = plt.subplots(1,2, figsize = (7.5,3.5))
            # for col in gen_data.columns:
            #     if col == "Perturb Factor":
            #         continue
            #     attack = col.split('_')[1]
            #     if "Acc" in col:
            #         sns.lineplot(gen_data, x = 'Perturb Factor', y = col, label = attack, ax = axes[0], marker=markers[attack], markersize = '8')
            #     elif "L2" in col:
            #         sns.lineplot(gen_data, x = 'Perturb Factor', y = col, label = attack, ax = axes[1], marker=markers[attack], markersize = '8')
            # axes[0].set_ylabel('Accuracy')
            # axes[0].set_yticks([0, 25, 50, 75, 100])
            # axes[1].set_ylabel('Mean $L_2$ Norm')
            # axes[0].ticklabel_format(axis='x', style='sci', scilimits=(2,3))
            # axes[1].ticklabel_format(axis='x', style='sci', scilimits=(2,3))
            # # axes[0].get_legend().remove()
            # axes[0].grid(True)
            # axes[1].grid(True)
            # axes[0].legend([])
            # axes[1].legend(title = "Attacks", bbox_to_anchor = (1,1))

            # fig.suptitle(f"Accuracy of model and $L_2$ norm  of perturbation \n under {attackType[evaltype].lower()} attacks in {xbox} setting")
            # plt.tight_layout()
            # plt.savefig(f"{plot_dir}/attack_gen_acc_norm_{xbox}_{env}_{cfg.see_max}.jpg", dpi = 300)
            # plt.savefig(f"{plot_dir}/attack_gen_acc_norm_{xbox}_{env}_{cfg.see_max}.pdf", dpi = 300)
            # plt.show(block=False)  
            # time.sleep(1)
            # plt.close()

            # Set the style of the plots
            sns.set(style="whitegrid")

            # Create a figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

            # Bar plot for Loss (L2)
            sns.barplot(ax=axes[1], data=gen_data, x='Attack', y='L2') #, palette='coolwarm')
            axes[1].set_title('Loss (L2) Across Attacks', fontsize=14)
            axes[1].set_xlabel('Attack Type')
            axes[1].set_ylabel('L2 Loss')

            # Bar plot for Accuracy (Acc)
            sns.barplot(ax=axes[0], data=gen_data, x='Attack', y='Acc') #, palette='viridis')
            axes[0].set_title('Accuracy Across Attacks', fontsize=14)
            axes[0].set_xlabel('Attack Type')
            axes[0].set_ylabel('Accuracy (%)')
            # Show the plot
            plt.savefig(f"{plot_dir}/attack_gen_acc_norm_{xbox}_{env}_{cfg.see_max}.jpg", dpi = 300)
            plt.savefig(f"{plot_dir}/attack_gen_acc_norm_{xbox}_{env}_{cfg.see_max}.pdf", dpi = 300)
            plt.show(block=False)  
            time.sleep(1)
            # plt.close()
            # NDSS Version: ------------------------
            #----------------------------------------
            #%%
            # Visualize tsne of the confidences
            results_gen_env_df = filter_dataset_by_evaltype(cfg, attack_analysis_df, col_dict)
            print(list(results_gen_env_df['Attack'].unique()))
            results_gen_df = results_gen_env_df[results_gen_env_df['Environment'] == env]

            tsne = TSNE(n_components=2, verbose=0, perplexity=10, n_iter=1000)
            tsne_results = tsne.fit_transform(results_gen_df[adv_noi_feat_col])
            results_gen_df['Component 1'] = tsne_results[:,0]
            results_gen_df['Component 2'] = tsne_results[:,1]

            # Plot the tsne graphs for different attacks and epsilons
            # eps_list = results_gen_df['Epsilon'].unique()
            attack_list = list(results_gen_df['Attack'].unique())
            print("attack_list unique: ", attack_list)
            attack_list = [attack for attack in attack_list if attack != 'RN']
            fig, axes = plt.subplots(1, len(attack_list), figsize = (len(attack_list)*2,3), sharey=True)
            # eps = cfg.dataset.epsilons[-1]
            print("attack_list :", attack_list )
            print("len(attack_list) : ", len(attack_list), attack_list)
            for ax, attack in zip(axes.flatten(), attack_list):

                # Load natural and adversarial images.....
                # data_dir = file_path(f"{cfg.results_dir}/data/{attack}/attack_data_ben_images_{eps}_{env}_{cfg.blackbox}.npy")
                # data_dir = file_path(f"{cfg.results_dir}/data/{attack}/attack_data_Benign_Org_Image_{env}_{cfg.blackbox}.npy")
                # ben_images = np.load(data_dir)
                # ben_images = ben_images.reshape(ben_images.shape[0], -1)
                
                # # data_dir = file_path(f"{cfg.results_dir}/data/{attack}/attack_data_adv_images_{eps}_{env}_{cfg.blackbox}.npy")
                # data_dir = file_path(f"{cfg.results_dir}/data/{attack}/attack_data_Adversarial_Org_Image_{env}_{cfg.blackbox}.npy")
                # adv_images = np.load(data_dir) 
                # adv_images = adv_images.reshape(adv_images.shape[0], -1)
                filter_1 = results_gen_df['Attack'] == attack
                filter_2 = results_gen_df['Attack'] == 'RN'
                data = results_gen_df.where(filter_1 | filter_2).dropna()
                print("data : ", data)
                data = data.replace(attack, 'Attack')
                data = data.replace("RN", 'Benign')

                # data = data[data['Epsilon'].isin(eps_list)]
                # data = data.groupby('Epsilon', group_keys=False).apply(lambda x: x.sample(125, replace=True))
                deep_pal = sns.color_palette('deep')
                palette = sns.blend_palette([deep_pal[2],  deep_pal[1], deep_pal[3]], 2)
                sns.scatterplot(
                    x="Component 1", y="Component 2",
                    hue="Attack",
                    palette = palette,
                    data = data,
                    legend="full",
                    edgecolor='black', linewidth=0.5,
                    alpha=0.99, ax = ax)
                ax.set_title(f"{attack} Attack")
                ax.get_legend().remove()
            ax.legend(loc = 'center', bbox_to_anchor = (1.35,0.5))
            fig.suptitle(f"T-SNE of the reconstruction noises for {attackType[evaltype].lower()} attacks at {cfg.rep}-level in {xbox} setting", fontsize = '13')
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/tsne_dist_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.jpg", dpi = 300)
            plt.savefig(f"{plot_dir}/tsne_dist_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.pdf", dpi = 300)
            plt.show(block=False)  
            time.sleep(1)
            plt.close()



            # #----------------------------------------
            # #%%
            # # Visualize tsne of the confidences
            # results_gen_env_df = filter_dataset_by_evaltype(cfg, attack_analysis_df, col_dict)
            # print(list(results_gen_env_df['Attack'].unique()))
            # results_gen_df = results_gen_env_df[results_gen_env_df['Environment'] == env]

            # tsne = TSNE(n_components=2, verbose=0, perplexity=10, n_iter=1000)
            # tsne_results = tsne.fit_transform(results_gen_df[adv_noi_feat_col])
            # results_gen_df['Component 1'] = tsne_results[:,0]
            # results_gen_df['Component 2'] = tsne_results[:,1]

            # # Plot the tsne graphs for different attacks and epsilons
            # # eps_list = results_gen_df['Epsilon'].unique()
            # attack_list = list(results_gen_df['Attack'].unique())
            # print("attack_list unique: ", attack_list)
            # attack_list = [attack for attack in attack_list if attack != 'GN']
            # fig, axes = plt.subplots(1, len(attack_list), figsize = (len(attack_list)*2,3), sharey=True)

            # print("attack_list :", attack_list )
            # print("len(attack_list) : ", len(attack_list), attack_list)
            # for ax, attack in zip(axes.flatten(), attack_list):
        
            #     filter_1 = results_gen_df['Attack'] == attack
            #     filter_2 = results_gen_df['Attack'] == 'RN'
            #     data = results_gen_df.where(filter_1 | filter_2).dropna()
            #     print("data : ", data)
            #     data = data.replace(attack, 'Attack')
            #     data = data.replace("RN", 'Benign')

            #     # data = data[data['Epsilon'].isin(eps_list)]
            #     # data = data.groupby('Epsilon', group_keys=False).apply(lambda x: x.sample(125, replace=True))
            #     deep_pal = sns.color_palette('deep')
            #     palette = sns.blend_palette([deep_pal[2],  deep_pal[1], deep_pal[3]], 2)
            #     sns.scatterplot(
            #         x="Component 1", y="Component 2",
            #         hue="Epsilon",
            #         palette = palette,
            #         data = data,
            #         legend="full",
            #         edgecolor='black', linewidth=0.5,
            #         alpha=0.99, ax = ax)
            #     ax.set_title(f"{attack} Attack")
            #     ax.get_legend().remove()
            # ax.legend(title = "Epsilon", loc = 'center', bbox_to_anchor = (1.35,0.5))
            # fig.suptitle(f"T-SNE of the reconstruction noises for {attackType[evaltype].lower()} attacks at {cfg.rep}-level in {xbox} setting", fontsize = '13')
            # plt.tight_layout()
            # plt.savefig(f"{plot_dir}/tsne_dist_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.jpg", dpi = 300)
            # plt.savefig(f"{plot_dir}/tsne_dist_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.pdf", dpi = 300)
            # plt.show(block=False)  
            # time.sleep(1)
            # plt.close()

            #%%
            #-------------------------------------------------------------------
            #           Plot the ROC Curves \w AUROC Scores                    #
            #------------------------------------------------------------------- 
            results_df = pd.read_csv(f"{cfg.results_dir}/detection/final_scores_adversarial_{cfg.blackbox}_{evaltype}_{evalenv}_100_{n_components}.csv", index_col=0)
            print("results_df: ", results_df)
            print("Unique Attacks: ", results_df['Attack'].value_counts())

            print("\n\n", results_df['Attack'].value_counts(), "\n\n")
            
            # results_df = results_df[results_df['Attack'] != 'GN']
            # results_df = results_df[results_df['Environment'] == env]
            # print("\n\n", results_df['Attack'].value_counts(), "\n\n")
            # # plot attack vs detector vs epsilon vs auroc
            # plt.figure(figsize=(10,2.5))
            # sns.barplot(results_df, y = "AUROC",  x = "Detector", hue="Attack")
            # # sns.boxenplot(results_df, y = "AUROC",  x = "Detector", hue="Attack")
            # plt.grid(True)
            # plt.legend(bbox_to_anchor = (1,1))
            # plt.title(f"Overall performance against {attackType[evaltype].lower()} attacks at {cfg.rep}-level w.r.t. {evalenv} under {xbox} setting")
            # plt.tight_layout()
            # plt.savefig(f"{plot_dir}/all_detectors_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.jpg", dpi = 300)
            # plt.savefig(f"{plot_dir}/all_detectors_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.pdf", dpi = 300)
            # plt.show(block=False)  
            # time.sleep(1)
            # plt.close()
            # Best detector
            best_detector = results_df[['Detector', 'AUROC']].groupby('Detector').mean().sort_values(['AUROC'], ascending=False)


            # det_list = list(set(list(best_detector.index)[0:3] + ['MagNet_L1', 'MagNet_JSD']))
            det_list = list(set(list(best_detector.index)) - {'MagNet_L1', 'MagNet_JSD', 'NADL'})[0:5] + ['MagNet_L1', 'MagNet_JSD', 'NADL']
            # df = results_df[results_df['Detector'].isin(det_list)]
        
            # plt.figure()
            # sns.set_theme(style="whitegrid")
            # ax = sns.catplot(df,  col = "Attack", x = "Detector", y = "AUROC",  
            #              kind="point", height=3.00, aspect=1.0,) 
            # # plt.legend(bbox_to_anchor = (1,1))
            # plt.ylim(0.0, 1.05)
            # ax.fig.subplots_adjust(top=0.8) # adjust the Figure in rp
            # ax.fig.suptitle(f"Top Detectors (and baselines) against {attackType[evaltype].lower()} attacks at {cfg.rep}-level w.r.t. {evalenv} under {xbox} setting")
            # # plt.tight_layout()
            # plt.savefig(f"{plot_dir}/top_detectors_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.jpg", dpi = 300)
            # plt.savefig(f"{plot_dir}/top_detectors_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.pdf", dpi = 300)
            # plt.show()


            # det_list = list(set(list(best_detector.index)[0:3] + ['MagNet_L1', 'MagNet_JSD']))
            det_list = list(set(list(best_detector.index)) - {'MagNet_L1', 'MagNet_JSD', 'PCA'})
            df = results_df[results_df['Detector'].isin(det_list)]
            if cfg.blackbox:
                df=df[df['Attack']!='BadNet']
            plt.figure(figsize=(10,3.5))
            sns.barplot(df, y = "AUROC",  x = "Detector", hue="Attack")
            plt.grid(True)
            plt.legend(bbox_to_anchor = (1.0,1.05))
            # plt.title(f"Overall performance against {attackType[evaltype].lower()} attacks at {cfg.rep}-level w.r.t. {evalenv} under {xbox} setting")
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/noisec_detectors_only_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.jpg", dpi = 300)
            plt.savefig(f"{plot_dir}/noisec_detectors_only_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.pdf", dpi = 300)
            plt.show(block=False)  
            time.sleep(1)
            plt.close()

            # Adding for the NDSS submission:
            #=====================================================
            # det_list = list(set(list(best_detector.index)[0:3] + ['MagNet_L1', 'MagNet_JSD']))
            det_list = list(set(list(best_detector.index)) - {'MagNet_L1', 'MagNet_JSD', 'PCA'})
            df = results_df[results_df['Detector'].isin(det_list)]
            #if cfg.blackbox:
            df=df[df['Attack']!='BadNet']
            plt.figure(figsize=(10,3.5))
            sns.barplot(df, y = "AUROC",  x = "Detector", hue="Attack")
            plt.grid(True)
            plt.legend(bbox_to_anchor = (1.0,1.05))
            # plt.title(f"Overall performance against {attackType[evaltype].lower()} attacks at {cfg.rep}-level w.r.t. {evalenv} under {xbox} setting")
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/noisec_detectors_only_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.jpg", dpi = 300)
            plt.savefig(f"{plot_dir}/noisec_detectors_only_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.pdf", dpi = 300)
            plt.show(block=False)  
            time.sleep(1)
            plt.close()
            #======================================================



            # det_list = list(set(list(best_detector.index)[0:3] + ['MagNet_L1', 'MagNet_JSD']))
            det_list = list(set(list(best_detector.index)) - {'PCA'}) #, 'MagNet_JSD'}) # - {'MagNet_L1', 'MagNet_JSD'})
            df = results_df[results_df['Detector'].isin(det_list)]
            print("Unique Attacks: ", df['Attack'].value_counts())
            df.replace('KNN', 'NoiSec\n(KNN)', inplace=True)
            df.replace('GMM', 'NoiSec\n(GMM)', inplace=True)
            df.replace('iKNN', 'NoiSec\n(iKNN)', inplace=True)
            df.replace('iGMM', 'NoiSec\n(iGMM)', inplace=True)
            df.replace('Max', 'NoiSec\n(MAX)', inplace=True)
            df.replace('STD', 'NoiSec\n(STD)', inplace=True)
            df.replace('NADL', 'NoiSec\n(xAD)', inplace=True)
            df.replace('MagNet_L1', 'MagNet\n(L1)', inplace=True)
            df.replace('MagNet_JSD', 'MagNet\n(JSD)', inplace=True)

            #if cfg.blackbox:
            df=df[df['Attack']!='BadNet']
            plt.figure(figsize=(10.0, 2.5))
            ax = sns.barplot(df, y = "AUROC",  x = "Detector", hue="Attack")
            # Define hatches for 7 categories

            plt.xticks(fontsize=10)
            plt.grid(True)

            plt.legend(bbox_to_anchor = (1.05,1.1))
            plt.yticks([0.0, 0.25, 0.50, 0.75, 1.0])
            
            plt.title(f"Overall performance against {attackType[evaltype].lower()} attacks under {xbox} setting")
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/noisec_detectors_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.jpg", dpi = 300)
            plt.savefig(f"{plot_dir}/noisec_detectors_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.pdf", dpi = 300)
            plt.show(block=False)  
            time.sleep(1)
            plt.close()
            print(f"{plot_dir}/noisec_detectors_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.pdf")

            # Adding for the NDSS submission:
            #=====================================================
            # det_list = list(set(list(best_detector.index)[0:3] + ['MagNet_L1', 'MagNet_JSD']))
            det_list = list(set(list(best_detector.index)) - {'PCA'}) #, 'MagNet_JSD'}) # - {'MagNet_L1', 'MagNet_JSD'})
            df = results_df[results_df['Detector'].isin(det_list)]
            df.replace('KNN', 'NoiSec(KNN-L2)', inplace=True)
            df.replace('KNN_L1', 'NoiSec(KNN-L1)', inplace=True)
            df.replace('GMM', 'NoiSec(GMM)', inplace=True)
            df.replace('iKNN', 'NoiSec(iKNN)', inplace=True)
            df.replace('iGMM', 'NoiSec(iGMM)', inplace=True)
            df.replace('Max', 'NoiSec(Max)', inplace=True)
            df.replace('Std', 'NoiSec(Std)', inplace=True)
            df.replace('NADL', 'NoiSec(NADL)', inplace=True)
            df.replace('MagNet_L1', 'MagNet(L1)', inplace=True)
            df.replace('MagNet_JSD', 'MagNet(JSD)', inplace=True)

            if cfg.blackbox:
                df=df[df['Attack']!='BadNet']

            # plt.figure(figsize=(10,4))
            # sns.lineplot(df, y = "AUROC",  x = "Detector", hue="Attack", style="Attack", linewidth=0.1, markers=True, markersize=8, alpha=0.25)
            # # Rotate the x-axis labels by 45 degrees
            # plt.xticks(rotation=45)
            # plt.grid(True)
            # plt.legend(bbox_to_anchor = (1.0,1.10))
            # # plt.yticks([0.0, 0.25, 0.50, 0.75, 1.0])
            # # plt.title(f"Overall performance against {attackType[evaltype].lower()} attacks at {cfg.rep}-level w.r.t. {evalenv} under {xbox} setting")
            # plt.tight_layout()
            # plt.savefig(f"{plot_dir}/noisec_detectors_performance_auroc_line_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.jpg", dpi = 300)
            # plt.savefig(f"{plot_dir}/noisec_detectors_performance_auroc_line_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.pdf", dpi = 300)
            # plt.show(block=False)  
            # time.sleep(1)
            # plt.close()
            #======================================================

            #%%
            # Plot ROC Curves
            dataset_name = cfg.dataset.name
            roc_curve_df = pd.read_csv(f"{cfg.results_dir}/detection/final_roc_curves_adversarial_{cfg.blackbox}_{evaltype}_{evalenv}_100_{n_components}.csv", index_col=0)
            roc_curve_df = roc_curve_df[roc_curve_df['Attack'] != 'GN']
            if cfg.blackbox:
                roc_curve_df=roc_curve_df[roc_curve_df['Attack']!='BadNet']
            roc_curve_df = roc_curve_df[roc_curve_df['Environment'] == env]

            # Added for ESORICS 2024-----------------------------------------
                
            # Plot ROC by epsion
            for detector in roc_curve_df['Detector'].unique():
                df_det = roc_curve_df[roc_curve_df['Detector'] == detector]
                # plt.figure(figsize=(3.5,2.75))
                plt.figure(figsize=(4,3))

                for indx, attack in enumerate(df_det['Attack'].unique()):
                    df_att = df_det[df_det['Attack'] == attack]
                    # for eps in df_att['Epsilon'].unique():
                    # df_eps = df_att[df_att['Epsilon'] == eps]
                    df_eps = df_att #[df_att['Epsilon'] == eps]
                    auroc = results_df.loc[
                        # (results_df['Epsilon'] == eps) & 
                        (results_df['Attack'] == attack) & 
                        (results_df['Detector'] == detector)]['AUROC']
                    # print("auroc : ", auroc)
                    auroc = auroc.values[0]
                    # sns.scatterplot(df_eps, x = "FPR", y = 'TPR') #, label = '{:.3f} : {:0.3f}'.format(eps, auroc))
                    plt.plot(df_eps["FPR"], df_eps['TPR'], marker = markers[attack], markersize = '3.5', label = '{}: {:0.3f}'.format(attack, auroc))
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                plt.legend(loc = 'lower right') #,  fontsize = '11')
                # plt.title(f"{attackType[evaltype]} {attack} attacks at {cfg.rep}-level \n w.r.t. {evalenv} with {detector} under {xbox} setting")
                plt.tight_layout()
                plt.savefig(f"{plot_dir}/roc_curves/roc_curve_all_attacks_{dataset_name}_{detector}_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.jpg", dpi = 300)
                plt.savefig(f"{plot_dir}/roc_curves/roc_curve_all_attacks_{dataset_name}_{detector}_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.pdf", dpi = 300)
                plt.show(block=False)  
                time.sleep(1)
                plt.close()

            #-----------------------------------------------------------------
            # Plot ROC by epsion
            for detector in roc_curve_df['Detector'].unique():
                if detector not in det_list[0:3]:
                    continue
                df_det = roc_curve_df[roc_curve_df['Detector'] == detector]
                
                for attack in df_det['Attack'].unique():
                    df_att = df_det[df_det['Attack'] == attack]
                    plt.figure(figsize=(4,3))
                    # for eps in df_att['Epsilon'].unique():
                    # df_eps = df_att[df_att['Epsilon'] == eps]
                    df_eps = df_att #[df_att['Epsilon'] == eps]
                    auroc = results_df.loc[
                        # (results_df['Epsilon'] == eps) & 
                        (results_df['Attack'] == attack) & 
                        (results_df['Detector'] == detector)]['AUROC']
                    # print("auroc : ", auroc)
                    auroc = auroc.values[0]
                    # sns.scatterplot(df_eps, x = "FPR", y = 'TPR') #, label = '{:.3f} : {:0.3f}'.format(eps, auroc))
                    plt.plot(df_eps["FPR"], df_eps['TPR'], marker = '.', label = '{:0.3f}'.format(auroc))
                    plt.legend(title = 'Epsilon : AUROC', loc = 'lower right') #,  fontsize = '11')
                    plt.title(f"{attackType[evaltype]} {attack} attacks at {cfg.rep}-level \n w.r.t. {evalenv} with {detector} under {xbox} setting")
                    plt.tight_layout()
                    plt.savefig(f"{roc_dir}/roc_curve_diff_eps_{attack}_{detector}_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.jpg", dpi = 300)
                    plt.savefig(f"{roc_dir}/roc_curve_diff_eps_{attack}_{detector}_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.pdf", dpi = 300)
                    plt.show(block=False)  
                    time.sleep(1)
                    plt.close()

            print(f"{roc_dir}/roc_curve_diff_eps_{attack}_{detector}_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.pdf")
            #%%
            # Plot ROC by detectors
            # for eps in roc_curve_df['Epsilon'].unique():
            df_eps = roc_curve_df #[roc_curve_df['Epsilon'] == eps]
            for attack in df_eps['Attack'].unique():
                df_att = df_eps[df_eps['Attack'] == attack]
                # plt.figure(figsize=(4,3))
                plt.figure(figsize=(4,3))
                for detector in df_att['Detector'].unique():
                    print("detector_name_dict[detector] : ", detector_name_dict[detector])
                    if detector not in det_list:
                        continue
                    df_det = df_att[df_att['Detector'] == detector]
                    auroc = results_df.loc[
                        # (results_df['Epsilon'] == eps) & 
                        (results_df['Attack'] == attack) & 
                        (results_df['Detector'] == detector)]['AUROC'].values[0]
                    plt.plot(df_det["FPR"], df_det['TPR'], marker = '.', label = '{} : {:0.2f}'.format(detector_name_dict[detector], auroc))
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                plt.legend(loc = 'lower right') 
                plt.title(f"{dataset_name_dict[cfg.dataset.name]} Dataset")
                plt.tight_layout()
                plt.savefig(f"{roc_dir}/roc_curve_diff_det_{attack}_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.jpg", dpi = 300)
                plt.savefig(f"{roc_dir}/roc_curve_diff_det_{attack}_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.pdf", dpi = 300)
                plt.show(block=False)  
                time.sleep(1)
                plt.close()
                print(f"{roc_dir}/roc_curve_diff_det_{attack}_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.pdf")
                # #%%
                # #-------------------------------------------------------------------
                # #           Plot the AUROC Scores at Difference Confidence         
                # #------------------------------------------------------------------- 
                # results_df = pd.read_csv(f"{cfg.results_dir}/detection/final_scores_adversarial_{cfg.blackbox}_{evaltype}_{evalenv}_{cfg.p_step}.csv", index_col=0)
                # print("\n\n", results_df['Attack'].value_counts(), "\n\n")
                
                # results_df = results_df[results_df['Attack'] != 'GN']
                # results_df = results_df[results_df['Environment'] == env]
                # df = results_df[results_df['Detector'].isin(det_list)]

                # # plt.figure()
                # sns.set_theme(style="whitegrid")
                # ax = sns.catplot(df,  col = "Attack", x = "P_max_ben", y = "AUROC",  
                #             hue = "Detector", kind="point", height=3.00, aspect=1.0,) 
                # plt.ylim(0.0, 1.05)
                # ax.fig.subplots_adjust(top=0.8) # adjust the Figure in rp
                # ax.fig.suptitle(f"Top Detectors (and baselines) against {attackType[evaltype].lower()} attacks at {cfg.rep}-level w.r.t. {evalenv} under {xbox} setting")
                # # plt.tight_layout()
                # plt.savefig(f"{plot_dir}/top_confidence_ben_auroc_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.jpg", dpi = 300)
                # plt.savefig(f"{plot_dir}/top_confidence_ben_auroc_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.pdf", dpi = 300)
                # print(f"{plot_dir}/top_confidence_ben_auroc_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.pdf")
                # plt.show()

                # #------------------------------------------------------------------- 
                # results_df = pd.read_csv(f"{cfg.results_dir}/detection/final_scores_adversarial_{cfg.blackbox}_{evaltype}_{evalenv}_{cfg.p_step}.csv", index_col=0)
                # print("\n\n", results_df['Attack'].value_counts(), "\n\n")
                
                # results_df = results_df[results_df['Attack'] != 'GN']
                # results_df = results_df[results_df['Environment'] == env]
                # df = results_df[results_df['Detector'].isin(det_list)]

                # # plt.figure()
                # sns.set_theme(style="whitegrid")
                # ax = sns.catplot(df,  col = "Attack", x = "P_max_adv", y = "AUROC",  
                #             hue = "Detector", kind="point", height=3.00, aspect=1.0,) 
                # plt.ylim(0.0, 1.05)
                # ax.fig.subplots_adjust(top=0.8) # adjust the Figure in rp
                # ax.fig.suptitle(f"Top Detectors (and baselines) against {attackType[evaltype].lower()} attacks at {cfg.rep}-level w.r.t. {evalenv} under {xbox} setting")
                # # plt.tight_layout()
                # plt.savefig(f"{plot_dir}/top_confidence_adv_auroc_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.jpg", dpi = 300)
                # plt.savefig(f"{plot_dir}/top_confidence_adv_auroc_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.pdf", dpi = 300)
                # print(f"{plot_dir}/top_confidence_adv_auroc_{xbox}_{env}_{evaltype}_{evalenv}_{cfg.see_max}.pdf")
                # plt.show()
                
            
                #-------------------------------------------------------------------------------------------------
                #                                    Adaptive Attacker
                #-------------------------------------------------------------------------------------------------

                # Results of the attack generation
                if cfg.adaptive:
    
                
                    list_of_files = glob.glob(f"{cfg.results_dir}*/data/attack_analysis_df_{dataset}_{cfg.blackbox}.csv")
                    if len(list_of_files) == 0:
                        print(f"No such file in {cfg.results_dir}*/data/attack_analysis_df_{dataset}_{cfg.blackbox}.csv")
                        return
                    attack_adaptive_results = pd.DataFrame([])
                    for attack_analysis_dir in list_of_files:
                        df = pd.read_csv(attack_analysis_dir, index_col=0)
                        attack_adaptive_results = pd.concat([attack_adaptive_results, df], axis=0)
                    attack_adaptive_results = attack_adaptive_results[attack_adaptive_results['Environment'] == env]

                    # Create table for L-2 norm and Accuracy
                    attack_adaptive_df = pd.DataFrame()

                    for eps in attack_adaptive_results['AdaptiveEps'].unique():
                        # for eps in results_gen_df['Epsilon'].unique():
                        filter_1 = attack_adaptive_results['AdaptiveEps'] == eps
                        df = attack_adaptive_results.where(filter_1)[adv_img_preds_col+org_targets_col+adv_per_norm_col].dropna()
                        y_pred = df[adv_img_preds_col].astype(int)
                        y_true = df[org_targets_col].astype(int)
                        accuracy = accuracy_score(y_true, y_pred)
                        attack_adaptive_df.loc[eps, f"Norm (L-2)"] = float(round(df[adv_per_norm_col].mean(), 2))
                        attack_adaptive_df.loc[eps, f"ASR"] = 100 - float(round(accuracy*100, 3))
                    attack_adaptive_df.insert(0, 'AdaptiveEps', attack_adaptive_df.index)
                    attack_adaptive_df = attack_adaptive_df.sort_values(['AdaptiveEps'])
                    print(attack_adaptive_df)


                    # Results of the attack generation
                    list_of_files = glob.glob(f"{cfg.results_dir}*/detection/final_scores_adversarial_{cfg.blackbox}_{evaltype}_{evalenv}_100_{n_components}.csv")
                    detect_adaptive_df = pd.DataFrame([])
                    for detect_analysis_dir in list_of_files:
                        df = pd.read_csv(detect_analysis_dir, index_col=0)
                        detect_adaptive_df = pd.concat([detect_adaptive_df, df], axis=0)
                    detect_adaptive_df = detect_adaptive_df[detect_adaptive_df['Environment'] == env]
                    print(detect_adaptive_df['Attack'].value_counts())
                    print(detect_adaptive_df['AdaptiveEps'].value_counts())
                    detect_adaptive_df = detect_adaptive_df[detect_adaptive_df['Attack'] == cfg.adaptive_attack]



                    # Plot the performance of adaptive attacks
                    # Create a figure and axis
                    df = attack_adaptive_df.copy()
                    fig, ax1 = plt.subplots(figsize= (4,3))
                    # Plot Norm (L-2) on the primary y-axis
                    line1, = ax1.plot(df["AdaptiveEps"], df["Norm (L-2)"], 'b-o', label="Norm (L-2)")
                    ax1.set_xlabel("Epsilon")
                    ax1.set_ylabel("Norm (L-2)", color='b')
                    ax1.tick_params(axis='y', labelcolor='b')

                    # Create a secondary y-axis for ASR
                    ax2 = ax1.twinx()
                    line2, = ax2.plot(df["AdaptiveEps"], df["ASR"], 'r-s', label="ASR")
                    ax2.set_ylabel("ASR", color='r')
                    ax2.tick_params(axis='y', labelcolor='r')

                    # Combine legends
                    lines = [line1, line2]
                    labels = [line.get_label() for line in lines]
                    plt.legend(lines, labels, loc="upper left")

                    # Add a legend and grid
                    fig.tight_layout()
                    plt.title("Norm (L-2) and ASR vs Epsilon")
                    ax1.grid()

                    plt.tight_layout()
                    plt.savefig(f"{plot_dir}/adaptive_attack_acc_norm_{cfg.adaptive_attack}_{xbox}_{env}_{cfg.see_max}.jpg", dpi = 300)
                    plt.savefig(f"{plot_dir}/adaptive_attack_acc_norm_{cfg.adaptive_attack}_{xbox}_{env}_{cfg.see_max}.pdf", dpi = 300)
                    plt.show(block=False)  
                    time.sleep(1)
                    plt.close()
                    #--------------------------------------

                    # Plot the performance of adaptive attacks
                    # Create a figure and axis
                    df = detect_adaptive_df.copy()
                    sns.color_palette('viridis')
                    df.replace(replacement_dict, inplace=True)
                    cmap = cm.get_cmap("viridis")  # Use any matplotlib colormap you like
                    fig, ax = plt.subplots(figsize= (7,3))
                    sns.barplot(data=df, x = 'Detector', y = 'AUROC', hue = 'AdaptiveEps',  palette = cmap) # style = 'AdaptiveEps', markers = True,
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f"{plot_dir}/adaptive_attack_auroc_{cfg.adaptive_attack}_{xbox}_{env}_{cfg.see_max}.jpg", dpi = 300)
                    plt.savefig(f"{plot_dir}/adaptive_attack_auroc_{cfg.adaptive_attack}_{xbox}_{env}_{cfg.see_max}.pdf", dpi = 300)
                    plt.show(block=False)  
                    time.sleep(1)
                    plt.close()        
                    #=========================================================================================================   

# Main function
if __name__ == '__main__':
    anayze_results()

# %%


