
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

    # dataset = cfg.dataset.name
    # epsilons = cfg.dataset.epsilons
    # classes = cfg.dataset.classes

    attackType = cfg.attackType
    markers = cfg.markers
    evaltype = cfg.evaltype
    evalenv = cfg.evalenv

    dataset_name_dict = {'mnist' : 'MNIST',
                     'fashion' : 'F-MNIST',
                     'cifar10' : 'CIFAR-10',
                     'gtsrb' : 'GTSRB',
                     'speech' : 'SPEECHCOMMAND',
                     'cancer' : 'CANCER',
                     }
    detector_name_dict ={'Artifacts' : 'Artifact',
                         'MagNet_L1' : 'MagNet(L1)',
                         'MagNet_JSD' : 'MagNet(JSD)',
                         'KNN' : 'NoiSec(KNN-1)',
                         'KNN_L1' : 'NoiSec(KNN)',
                         'KNN_Cos' : 'NoiSec(KNNC)',
                         'GMM' : 'NoiSec(GMM)',
                         'iKNN' : 'NoiSec(iKNN)',
                         'iGMM' : 'NoiSec(iGMM)',
                         'Max' : 'NoiSec(Max)',
                         'STD' : 'NoiSec(Std)',
                         'Manda' : 'Manda',
                         }
    
    det_list = ['KNN_L1', 'GMM', 'STD', 'Max']

    with open(f"{cfg.results_dir}/data/col_dict.json", 'r') as file:
        col_dict = json.load(file)
    
    adv_img_preds_col = col_dict['Adversarial_Org_Image']['Pred']
    adv_noi_feat_col = col_dict['Adversarial_Rec_Noise'][cfg.rep]
    adv_per_norm_col = col_dict['Adversarial_Org_Noise']['Norm']
    org_targets_col = ['Target']

        
    for env in cfg.environments:
        print(f"Environment: {env}")
        xbox = 'Black-box' if cfg.blackbox else 'White-box'
        plot_dir = Path(f'{cfg.results_dir}/plots/Overall/{xbox}')

     
        #%%
        #-------------------------------------------------------------------
        #           Plot the ROC Curves \w AUROC Scores                    #
        #------------------------------------------------------------------- 
        results_df = pd.read_csv(f"{cfg.results_dir}/detection/final_scores_backdoor_{cfg.blackbox}_{evaltype}_{evalenv}_100.csv", index_col=0)
        results_df = results_df.replace('LabelConsistent', 'LCA')

        print("List of detector: ", results_df["Detector"].unique())
        print("List of Attack: ", results_df["Attack"].unique())
        print("results_df: ", results_df)
        print("\n\n", results_df['Attack'].value_counts(), "\n\n")
        
        results_df = results_df[results_df['Attack'] != 'GN']
        results_df = results_df[results_df['Environment'] == env]
        print("\n\n", results_df['Attack'].value_counts(), "\n\n")
        # plot attack vs detector vs epsilon vs auroc
        plt.figure(figsize=(10,2.5))
        sns.barplot(results_df, y = "AUROC",  x = "Detector", hue="Attack")
        # sns.boxenplot(results_df, y = "AUROC",  x = "Detector", hue="Attack")
        plt.grid(True)
        plt.legend(bbox_to_anchor = (1,1))
        plt.title(f"Overall performance against {attackType[evaltype].lower()} attacks at {cfg.rep}-level w.r.t. {evalenv} under {xbox} setting")
        plt.tight_layout()
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{plot_dir}/backdoor_all_detectors_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}.jpg", dpi = 300)
        plt.savefig(f"{plot_dir}/backdoor_all_detectors_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}.pdf", dpi = 300)
        plt.show(block=False)  
        time.sleep(1)
        plt.close()
        # Best detector
        best_detector = results_df[['Detector', 'AUROC']].groupby('Detector').mean().sort_values(['AUROC'], ascending=False)


        # # det_list = list(set(list(best_detector.index)[0:3] + ['MagNet_L1', 'MagNet_JSD']))
        # det_list = list(set(list(best_detector.index)) - {'MagNet_L1', 'MagNet_JSD'})[0:5] + ['MagNet_L1', 'MagNet_JSD']
        # # det_list = ['KNN_L1', 'PCA', 'GMM', 'Std', 'MagNet_L1', 'MagNet_JSD']
        df = results_df[results_df['Detector'].isin(det_list)]

        # plt.figure()
        sns.set_theme(style="whitegrid")
        ax = sns.catplot(df,  col = "Attack", x = "Epsilon", y = "AUROC",  
                    hue = "Detector", kind="point", height=3.00, aspect=1.0,) 
        # plt.legend(bbox_to_anchor = (1,1))
        plt.ylim(0.0, 1.05)
        ax.fig.subplots_adjust(top=0.8) # adjust the Figure in rp
        ax.fig.suptitle(f"Top Detectors (and baselines) against {attackType[evaltype].lower()} attacks at {cfg.rep}-level w.r.t. {evalenv} under {xbox} setting")
        # plt.tight_layout()
        plt.savefig(f"{plot_dir}/backdoor_top_detectors_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}.jpg", dpi = 300)
        plt.savefig(f"{plot_dir}/backdoor_top_detectors_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}.pdf", dpi = 300)
        plt.show()


        # # det_list = list(set(list(best_detector.index)[0:3] + ['MagNet_L1', 'MagNet_JSD']))
        # # det_list = list(set(list(best_detector.index)) - {'MagNet_L1', 'MagNet_JSD', 'PCA'})
        df = results_df[results_df['Detector'].isin(det_list)]
        if cfg.blackbox:
            df=df[df['Attack']!='BadNet']
        plt.figure(figsize=(4.75,2.5))
        sns.barplot(df, y = "AUROC",  x = "Detector", hue="Attack")
        plt.grid(True)
        plt.legend(bbox_to_anchor = (1.0,1.05))
        # plt.title(f"Overall performance against {attackType[evaltype].lower()} attacks at {cfg.rep}-level w.r.t. {evalenv} under {xbox} setting")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/backdoor_noisec_detectors_only_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}.jpg", dpi = 300)
        plt.savefig(f"{plot_dir}/backdoor_noisec_detectors_only_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}.pdf", dpi = 300)
        plt.show(block=False)  
        time.sleep(1)
        plt.close()

        # Adding for the NDSS submission:
        #=====================================================
        # # det_list = list(set(list(best_detector.index)[0:3] + ['MagNet_L1', 'MagNet_JSD']))
        # # det_list = list(set(list(best_detector.index)) - {'MagNet_L1', 'MagNet_JSD', 'PCA'})
        df = results_df[results_df['Detector'].isin(det_list)]
        #if cfg.blackbox:
        df=df[df['Attack']!='BadNet']
        plt.figure(figsize=(4.75,2.5))
        sns.barplot(df, y = "AUROC",  x = "Detector", hue="Attack")
        plt.grid(True)
        plt.legend(bbox_to_anchor = (1.0,1.05))
        # plt.title(f"Overall performance against {attackType[evaltype].lower()} attacks at {cfg.rep}-level w.r.t. {evalenv} under {xbox} setting")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/backdoor_noisec_detectors_only_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}.jpg", dpi = 300)
        plt.savefig(f"{plot_dir}/backdoor_noisec_detectors_only_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}.pdf", dpi = 300)
        plt.show(block=False)  
        time.sleep(1)
        plt.close()
        #======================================================



        # # det_list = list(set(list(best_detector.index)[0:3] + ['MagNet_L1', 'MagNet_JSD']))
        # det_list = list(set(list(best_detector.index)) - {'PCA'}) #, 'MagNet_JSD'}) # - {'MagNet_L1', 'MagNet_JSD'})
        df = results_df[results_df['Detector'].isin(det_list)]
        
        df.replace('KNN_L1', 'NoiSec\n(KNN)', inplace=True)
        df.replace('GMM', 'NoiSec\n(GMM)', inplace=True)
        df.replace('iKNN', 'NoiSec\n(iKNN)', inplace=True)
        df.replace('iGMM', 'NoiSec\n(iGMM)', inplace=True)
        df.replace('Max', 'NoiSec\n(MAX)', inplace=True)
        df.replace('STD', 'NoiSec\n(STD)', inplace=True)

        df.replace('MagNet_L1', 'MagNet\n(L1)', inplace=True)
        df.replace('MagNet_JSD', 'MagNet\n(JSD)', inplace=True)

        #if cfg.blackbox:
        df=df[df['Attack']!='BadNet']
        plt.figure(figsize=(7.0,2.0))
        ax = sns.barplot(df, y = "AUROC",  x = "Detector", hue="Attack")
        # Define hatches for 7 categories

        # # Function to determine font size based on bar width
        # def get_font_size(bar_width):
        #     # Adjust the multiplier as needed for desired font size
        #     return bar_width * 30

        # # Add values on top of each bar
        # for p in ax.patches:
        #     bar_width = p.get_width()
        #     font_size = get_font_size(bar_width)
        #     ax.annotate(format(p.get_height(), '.2f'),
        #                     (p.get_x() + bar_width / 2., p.get_height()),
        #                     ha = 'center', va = 'center',
        #                     xytext = (0,5),
        #                     textcoords = 'offset points',
        #                     fontsize=font_size,
        #                     rotation=90)  # Rotate the text vertically


        # # Define hatches for 7 categories (enough to cover all possible hues)
        # hatches = ['/', '\\', '|', '-', '+', 'x', '*']

        # # Get unique hue categories
        # unique_hues = df['Attack'].unique()

        # # Iterate over each hue category and apply hatches
        # for i, hue in enumerate(unique_hues):
        #     # Filter data for the current hue
        #     hue_data = df[df['Attack'] == hue]
            
        #     # Iterate over bars for the current hue and apply hatch
        #     for j, patch in enumerate(ax.patches[i::len(unique_hues)]):
        #         hatch_index = j % len(hatches)
        #         patch.set_hatch(hatches[hatch_index])

            
        # ax.xaxis.set_tick_params(width=1, length=5)  # Adjust the length as needed
        # Rotate the x-axis labels by 45 degrees
        # plt.xticks(rotation=45)
        plt.xticks(fontsize=10)
        plt.grid(True)

        # # Manually create legend entries with hatches
        # legend_handles = []
        # for i, hue in enumerate(unique_hues):
        #     # Create patch with color and hatch for each hue
        #     patch = plt.Rectangle((0, 0), 1, 1, color=ax.get_legend().legendHandles[i].get_facecolor(), hatch=hatches[i % len(hatches)])
        #     legend_handles.append(patch)

        # # Set legend with custom handles and labels
        # ax.legend(legend_handles, unique_hues, bbox_to_anchor=(1.0, 1.10), title='Attack', 
        #         handlelength=3, handleheight=1.5, handletextpad=1.0, loc='upper left')


        plt.legend(bbox_to_anchor = (1.05,1.1))
        plt.yticks([0.0, 0.25, 0.50, 0.75, 1.0])
        
        # plt.title(f"Overall performance against {attackType[evaltype].lower()} attacks at {cfg.rep}-level w.r.t. {evalenv} under {xbox} setting")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/backdoor_noisec_detectors_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}.jpg", dpi = 300)
        plt.savefig(f"{plot_dir}/backdoor_noisec_detectors_performance_auroc_{xbox}_{env}_{evaltype}_{evalenv}.pdf", dpi = 300)
        plt.show(block=False)  
        time.sleep(1)
        plt.close()

        # Adding for the NDSS submission:
        #=====================================================
        # # det_list = list(set(list(best_detector.index)[0:3] + ['MagNet_L1', 'MagNet_JSD']))
        # det_list = list(set(list(best_detector.index)) - {'PCA'}) #, 'MagNet_JSD'}) # - {'MagNet_L1', 'MagNet_JSD'})
        df = results_df[results_df['Detector'].isin(det_list)]
        df.replace('KNN_L1', 'NoiSec(KNN)', inplace=True)
        df.replace('GMM', 'NoiSec(GMM)', inplace=True)
        df.replace('iKNN', 'NoiSec(iKNN)', inplace=True)
        df.replace('iGMM', 'NoiSec(iGMM)', inplace=True)
        df.replace('Max', 'NoiSec(Max)', inplace=True)
        df.replace('Std', 'NoiSec(Std)', inplace=True)
        df.replace('MagNet_L1', 'MagNet(L1)', inplace=True)
        df.replace('MagNet_JSD', 'MagNet(JSD)', inplace=True)

        if cfg.blackbox:
            df=df[df['Attack']!='BadNet']

        plt.figure(figsize=(6,3))
        sns.lineplot(df, y = "AUROC",  x = "Detector", hue="Attack", style="Attack", linewidth=0.1, markers=True, markersize=8, alpha=0.25)
        # Rotate the x-axis labels by 45 degrees
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend(bbox_to_anchor = (1.0,1.10))
        # plt.yticks([0.0, 0.25, 0.50, 0.75, 1.0])
        # plt.title(f"Overall performance against {attackType[evaltype].lower()} attacks at {cfg.rep}-level w.r.t. {evalenv} under {xbox} setting")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/backdoor_noisec_detectors_performance_auroc_line_{xbox}_{env}_{evaltype}_{evalenv}.jpg", dpi = 300)
        plt.savefig(f"{plot_dir}/backdoor_noisec_detectors_performance_auroc_line_{xbox}_{env}_{evaltype}_{evalenv}.pdf", dpi = 300)
        plt.show(block=False)  
        time.sleep(1)
        plt.close()
        #======================================================

        #%%
        # Plot ROC Curves
        roc_curve_df = pd.read_csv(f"{cfg.results_dir}/detection/final_roc_curves_backdoor_{cfg.blackbox}_{evaltype}_{evalenv}_100.csv", index_col=0)
        roc_curve_df = roc_curve_df.replace("LabelConsistent", "LCA")
        roc_curve_df = roc_curve_df[roc_curve_df['Attack'] != 'GN']
        if cfg.blackbox:
            roc_curve_df=roc_curve_df[roc_curve_df['Attack']!='BadNet']
        roc_curve_df = roc_curve_df[roc_curve_df['Environment'] == env]

        # Added for ESORICS 2024-----------------------------------------
            
        # Plot ROC by epsion
        for detector in roc_curve_df['Detector'].unique():
            df_det = roc_curve_df[roc_curve_df['Detector'] == detector]
            plt.figure(figsize=(3.5,2.75))
            for attack in df_det['Attack'].unique():
                df_att = df_det[df_det['Attack'] == attack]
                for eps in df_att['Epsilon'].unique():
                    df_eps = df_att[df_att['Epsilon'] == eps]
                    auroc = results_df.loc[(results_df['Epsilon'] == eps) 
                                & (results_df['Attack'] == attack)
                                & (results_df['Detector'] == detector)]['AUROC']
                    # print("auroc : ", auroc)
                    auroc = auroc.values[0]
                    # sns.scatterplot(df_eps, x = "FPR", y = 'TPR') #, label = '{:.3f} : {:0.3f}'.format(eps, auroc))
                    plt.plot(df_eps["FPR"], df_eps['TPR'], marker = '.', label = '{}: {:0.3f}'.format(attack, auroc))
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
            plt.legend(loc = 'lower right') #,  fontsize = '11')
            # plt.title(f"{attackType[evaltype]} {attack} attacks at {cfg.rep}-level \n w.r.t. {evalenv} with {detector} under {xbox} setting")
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/backdoor_roc_curve_all_attacks_{detector}_{xbox}_{env}_{evaltype}_{evalenv}.jpg", dpi = 300)
            plt.savefig(f"{plot_dir}/backdoor_roc_curve_all_attacks_{detector}_{xbox}_{env}_{evaltype}_{evalenv}.pdf", dpi = 300)
            plt.show(block=False)  
            time.sleep(1)
            plt.close()

        #-----------------------------------------------------------------
        # Plot ROC by epsion
        for detector in roc_curve_df['Detector'].unique():
            if detector not in det_list:
                continue
            df_det = roc_curve_df[roc_curve_df['Detector'] == detector]
            
            for attack in df_det['Attack'].unique():
                df_att = df_det[df_det['Attack'] == attack]
                plt.figure(figsize=(4,3))
                for eps in df_att['Epsilon'].unique():
                    df_eps = df_att[df_att['Epsilon'] == eps]
                    auroc = results_df.loc[(results_df['Epsilon'] == eps) 
                                & (results_df['Attack'] == attack)
                                & (results_df['Detector'] == detector)]['AUROC']
                    # print("auroc : ", auroc)
                    auroc = auroc.values[0]
                    # sns.scatterplot(df_eps, x = "FPR", y = 'TPR') #, label = '{:.3f} : {:0.3f}'.format(eps, auroc))
                    plt.plot(df_eps["FPR"], df_eps['TPR'], marker = '.', label = '{:.3f} : {:0.3f}'.format(eps, auroc))
                    plt.legend(title = 'Epsilon : AUROC', loc = 'lower right') #,  fontsize = '11')
                plt.title(f"{attackType[evaltype]} {attack} attacks at {cfg.rep}-level \n w.r.t. {evalenv} with {detector} under {xbox} setting")
                plt.tight_layout()
                plt.savefig(f"{plot_dir}/backdoor_roc_curve_diff_eps_{attack}_{detector}_{xbox}_{env}_{evaltype}_{evalenv}.jpg", dpi = 300)
                plt.savefig(f"{plot_dir}/backdoor_roc_curve_diff_eps_{attack}_{detector}_{xbox}_{env}_{evaltype}_{evalenv}.pdf", dpi = 300)
                plt.show(block=False)  
                time.sleep(1)
                plt.close()

        #%%
        # Plot ROC by detectors
        for eps in roc_curve_df['Epsilon'].unique():
            df_eps = roc_curve_df[roc_curve_df['Epsilon'] == eps]
            for attack in df_eps['Attack'].unique():
                df_att = df_eps[df_eps['Attack'] == attack]
                # plt.figure(figsize=(4,3))
                plt.figure(figsize=(3.75,2.75))
                for detector in df_att['Detector'].unique():
                    print("detector_name_dict[detector] : ", detector_name_dict[detector])
                    if detector not in det_list:
                        continue
                    df_det = df_att[df_att['Detector'] == detector]
                    auroc = results_df.loc[(results_df['Epsilon'] == eps) 
                                & (results_df['Attack'] == attack)
                                & (results_df['Detector'] == detector)]['AUROC'].values[0]
                    plt.plot(df_det["FPR"], df_det['TPR'], marker = '.', label = '{} : {:0.2f}'.format(detector_name_dict[detector], auroc))
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                plt.legend(loc = 'lower right') 
                # plt.title(f"{dataset_name_dict[cfg.dataset.name]} Dataset")
                plt.tight_layout()
                plt.savefig(f"{plot_dir}/backdoor_roc_curve_diff_det_{attack}_{eps}_{xbox}_{env}_{evaltype}_{evalenv}.jpg", dpi = 300)
                plt.savefig(f"{plot_dir}/backdoor_roc_curve_diff_det_{attack}_{eps}_{xbox}_{env}_{evaltype}_{evalenv}.pdf", dpi = 300)
                plt.show(block=False)  
                time.sleep(1)
                plt.close()

        #%%
        #-------------------------------------------------------------------
        #           Plot the AUROC Scores at Difference Confidence         
        #------------------------------------------------------------------- 
        results_df = pd.read_csv(f"{cfg.results_dir}/detection/final_scores_backdoor_{cfg.blackbox}_{evaltype}_{evalenv}_{cfg.p_step}.csv", index_col=0)
        results_df = results_df.replace("LabelConsistent", "LCA")
        print("\n\n", results_df['Attack'].value_counts(), "\n\n")
        
        results_df = results_df[results_df['Attack'] != 'GN']
        results_df = results_df[results_df['Environment'] == env]
        df = results_df[results_df['Detector'].isin(det_list)]

        # plt.figure()
        sns.set_theme(style="whitegrid")
        ax = sns.catplot(df,  col = "Attack", x = "P_max_ben", y = "AUROC",  
                    hue = "Detector", kind="point", height=3.00, aspect=1.0,) 
        plt.ylim(0.0, 1.05)
        ax.fig.subplots_adjust(top=0.8) # adjust the Figure in rp
        ax.fig.suptitle(f"Top Detectors (and baselines) against {attackType[evaltype].lower()} attacks at {cfg.rep}-level w.r.t. {evalenv} under {xbox} setting")
        # plt.tight_layout()
        plt.savefig(f"{plot_dir}/backdoor_top_confidence_ben_auroc_{xbox}_{env}_{evaltype}_{evalenv}.jpg", dpi = 300)
        plt.savefig(f"{plot_dir}/backdoor_top_confidence_ben_auroc_{xbox}_{env}_{evaltype}_{evalenv}.pdf", dpi = 300)
        print(f"{plot_dir}/backdoor_top_confidence_ben_auroc_{xbox}_{env}_{evaltype}_{evalenv}.pdf")
        plt.show()

        #------------------------------------------------------------------- 
        results_df = pd.read_csv(f"{cfg.results_dir}/detection/final_scores_backdoor_{cfg.blackbox}_{evaltype}_{evalenv}_{cfg.p_step}.csv", index_col=0)
        print("\n\n", results_df['Attack'].value_counts(), "\n\n")
        
        results_df = results_df[results_df['Attack'] != 'GN']
        results_df = results_df[results_df['Environment'] == env]
        df = results_df[results_df['Detector'].isin(det_list)]

        # plt.figure()
        sns.set_theme(style="whitegrid")
        ax = sns.catplot(df,  col = "Attack", x = "P_max_adv", y = "AUROC",  
                    hue = "Detector", kind="point", height=3.00, aspect=1.0,) 
        plt.ylim(0.0, 1.05)
        ax.fig.subplots_adjust(top=0.8) # adjust the Figure in rp
        ax.fig.suptitle(f"Top Detectors (and baselines) against {attackType[evaltype].lower()} attacks at {cfg.rep}-level w.r.t. {evalenv} under {xbox} setting")
        # plt.tight_layout()
        plt.savefig(f"{plot_dir}/backdoor_top_confidence_adv_auroc_{xbox}_{env}_{evaltype}_{evalenv}.jpg", dpi = 300)
        plt.savefig(f"{plot_dir}/backdoor_top_confidence_adv_auroc_{xbox}_{env}_{evaltype}_{evalenv}.pdf", dpi = 300)
        print(f"{plot_dir}/backdoor_top_confidence_adv_auroc_{xbox}_{env}_{evaltype}_{evalenv}.pdf")
        plt.show()
        
# Main function
if __name__ == '__main__':
    anayze_results()
