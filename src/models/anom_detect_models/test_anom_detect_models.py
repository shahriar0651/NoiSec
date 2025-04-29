from tqdm import tqdm
import pandas as pd
import json
import time
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import tsfel
from sklearn.preprocessing import MinMaxScaler


def extract_feat(detector, X_test):
    df = pd.DataFrame(X_test)
    if detector == 'Max': 
        return df.max(axis=1).values
    elif detector == 'Range': 
        return (df.max(axis=1) - df.min(axis=1)).values
    elif detector == 'Mean': 
        return df.mean(axis=1).values
    elif detector == 'STD': 
        return df.std(axis=1).values

def get_tsfel_feats(cfg, X_test):
    cfg_file = tsfel.get_features_by_domain("statistical")  # All statistical domain features will be extracted
    # Step 2: Toggle the 'use' variable for each feature
    for feature in cfg_file['statistical']:
        if feature in cfg.models.tsfel_feats:
            cfg_file['statistical'][feature]['use'] = 'yes'
        else:
            cfg_file['statistical'][feature]['use'] = 'no'
    print("\n\n\n", cfg_file)

    # TODO: Modify cfg to only extract the listed features
    X_feats_total = pd.DataFrame([])
    for X_train_row in tqdm(X_test):
        X_feats_row = tsfel.time_series_features_extractor(cfg_file, X_train_row, fs=50, window_size=len(X_train_row), verbose=False)    # Receives a time series sampled at 50 Hz, divides into windows of size 250 (i.e. 5 seconds) and extracts all features
        X_feats_row.columns = [col.replace("0_","") for col in X_feats_row.columns]
        X_feats_total = pd.concat([X_feats_total, X_feats_row], axis=0)
    return X_feats_total

def get_prediction(cfg, model, X_test, y_cls, detector):

    if detector in cfg.models.model_List:
        y_pred = model.decision_function(X_test)
    else:
        print(f"Detector : {detector}")
    y_pred = np.array(y_pred).reshape(-1, 1)
    num_samples = len(y_pred)//2
    y_pred = MinMaxScaler().fit(y_pred[0:num_samples]).transform(y_pred)
    return y_pred
    

def test_baseline(cfg, model, X_feat, y_pred, y_test, y_cls, env, attack, detector, p_max_ben=1.0, p_max_adv=1.0):
    
    max_fpr = cfg.dataset.max_fpr

    if detector in cfg.models.artifacts:
        det_type = 'Artifacts'
    elif detector in cfg.models.magnet_det:
        det_type = 'MagNet'
    elif detector in cfg.models.manda:
        det_type = 'Manda'
    elif detector in cfg.models.model_List:
        det_type = 'Model'
    else:
        print(f"Detector : {detector}")

    # Prediction data
    predictions_df = pd.DataFrame([])
    predictions_df['y_true'] = y_test
    predictions_df['y_pred'] = y_pred 
    predictions_df['Detector'] = detector
    # predictions_df['Epsilon'] = eps
    predictions_df['P_max_ben'] = p_max_ben
    predictions_df['P_max_adv'] = p_max_adv
    predictions_df['Environment'] = env
    predictions_df['Attack'] = attack
    predictions_df['DetType'] = det_type

    # ROC Curve data
    try:
        fpr, tpr, thshlds = roc_curve(y_test, y_pred)
    except:
        print(y_test.min(), y_test.max(), y_pred.min(), y_pred.max(), y_pred)
        
    roc_curve_df = pd.DataFrame(
        {"TPR" : tpr,
         "FPR" : fpr})
    roc_curve_df['P_max_ben'] = p_max_ben
    roc_curve_df['P_max_adv'] = p_max_adv
    roc_curve_df['Detector'] = detector
    # roc_curve_df['Epsilon'] = eps
    roc_curve_df['Environment'] = env
    roc_curve_df['Attack'] = attack
    roc_curve_df['DetType'] = det_type

    # PR Curve data
    prec, recall, _ = precision_recall_curve(y_test, y_pred)   
    pr_curve_df = pd.DataFrame({"Precision" : prec,
                                "Recall" : recall,
                                })
    pr_curve_df['P_max_ben'] = p_max_ben
    pr_curve_df['P_max_adv'] = p_max_adv
    pr_curve_df['Detector'] = detector
    # pr_curve_df['Epsilon'] = eps
    pr_curve_df['Environment'] = env
    pr_curve_df['Attack'] = attack
    pr_curve_df['DetType'] = det_type

    # Pre, Rec, FPR,, and F1 Score
    # Calculate FPR with maximum FPR
    thshlds_index = np.where(fpr<max_fpr)[0][-1] #next(i for i, fpr_val in enumerate(fpr) if fpr_val > max_fpr)
    thshlds_fpr = thshlds[thshlds_index]
    y_pred_cls = (y_pred >= thshlds_fpr)

    # Calculate Precision, Recall, and F1-score
    pre_th = precision_score(y_test, y_pred_cls)
    rec_th = recall_score(y_test, y_pred_cls)
    f1_th = f1_score(y_test, y_pred_cls)
    fpr_th = fpr[thshlds_index]


    # AUC data
    try:
        auroc_score = roc_auc_score(y_test, y_pred)
        auprc_score = average_precision_score(y_test, y_pred)
        print(f"detector : {detector} -->  AUROC: {auroc_score}, AUPRC: {auprc_score}")
    except ValueError as e:
        print(e, "Setting dummy score of 0.0") 
        auroc_score = 0.0
        auprc_score = 0.0
    auc_scores_df =  pd.Series({
        "Detector": detector,
        "DetType" : det_type,
        # "Epsilon": eps,   
        "P_max_ben" : p_max_ben,     
        "P_max_adv" : p_max_adv,     
        "Environment": env,        
        "Attack": attack,        
        "AUROC": round(auroc_score, 4), 
        "AUPRC": round(auprc_score, 4),
        "Precision" : round(auroc_score, 4), 
        "Recall" : round(rec_th, 4), 
        "FPR" : round(fpr_th, 4), 
        "F1Score" : round(f1_th, 4), 
        })

    # print(auc_scores_df)
    return predictions_df, roc_curve_df, pr_curve_df, auc_scores_df
        
