import subprocess
import numpy as np
import os
import pandas as pd
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from typing import List, Callable
from collections import defaultdict

import torch
from torch.utils import data
from tqdm.notebook import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize

import sklearn
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import average_precision_score
from sklearn.utils import resample 

import scipy
import scipy.stats

import sys
sys.path.append('../..')

import clip
from model import CLIP
from eval import * 
from zero_shot import * 

def evaluate_model(X_dir, y_dir, model_path, cxr_labels, alt_labels_dict=None):
        cxr_filepath = X_dir
        final_label_path = y_dir

        results_out_folder = './results'
        context_length = 77

        # templates list of positive and negative template pairs
        cxr_pair_templates = [("{}", "no {}")]

        cxr_results, y_pred = run_zero_shot(cxr_labels, cxr_pair_templates, model_path, cxr_filepath=cxr_filepath, final_label_path=final_label_path, alt_labels_dict=alt_labels_dict, softmax_eval=True, context_length=context_length, pretrained=True, use_bootstrap=True, cutlabels=True)
        return cxr_results, y_pred
    
def f1_mcc_bootstrap(y_pred, y_true, cxr_labels, best_p_vals, eval_func, n_samples=5000, label_idx_map=None): 
    '''
    This function will randomly sample with replacement 
    from y_pred and y_true then evaluate `n` times
    and obtain AUROC scores for each. 

    You can specify the number of samples that should be
    used with the `n_samples` parameter. 

    Confidence intervals will be generated from each 
    of the samples. 
    '''
    y_pred # (500, 14)
    y_true # (500, 14)

    idx = np.arange(len(y_true))

    boot_stats = []
    for i in tqdm(range(n_samples)): 
        sample = resample(idx, replace=True)
        y_pred_sample = y_pred[sample]
        y_true_sample = y_true[sample]

        sample_stats = eval_func(y_pred_sample, y_true_sample, best_p_vals, cxr_labels=cxr_labels, label_idx_map=label_idx_map)
        boot_stats.append(sample_stats)

    boot_stats = pd.concat(boot_stats) # pandas array of evaluations for each sample
    return boot_stats, compute_cis(boot_stats)

def get_best_alt_labels(res_df, cxr_labels): 
    best_alt_labels_dict = dict()
    best_alt_labels_vals = dict()
    res_cols = list(res_df)

    curr_path_name = None
    for col in res_cols: # for each col
        path_name = col.split("_")[0] # pathology name
        mean_auc = res_df[col][0] # mean auc

        if path_name in cxr_labels: 
            # reset the vars
            curr_path_name = path_name
            best_alt_labels_dict[path_name] = [path_name]
            best_alt_labels_vals[path_name] = mean_auc

        if best_alt_labels_vals[curr_path_name] < mean_auc: 
            best_alt_labels_vals[curr_path_name] = mean_auc
            best_alt_labels_dict[curr_path_name] = [path_name]

    return best_alt_labels_dict
    
def y_true_csv_to_np(df_path, cxr_labels): 
    groundtruth = pd.read_csv(df_path)
    groundtruth = groundtruth[cxr_labels]
    groundtruth = groundtruth.to_numpy()[:,:].astype(int)
    return groundtruth

def get_best_p_vals(pred, groundtruth, cxr_labels, metric_func=matthews_corrcoef, spline_k: int = None, verbose: bool = False):
    """
    WARNING: CXR_LABELS must 
    Params: 
    * pred : np arr
        probabilities output by model

    * plot_graphs : bool
        if True, will save plots for metric vs. threshold for 
        each pathology
        
    Note: 
    * `probabilities` value is a linspace of possible probabilities
    """
    probabilities = [val for val in np.arange(0.4, 0.64, 0.0001)]
    best_p_vals = dict()
    for idx, cxr_label in enumerate(cxr_labels):
        y_true = groundtruth[:, idx]
        _, _, probabilities = roc_curve(y_true, pred[:, idx])
        probabilities = probabilities[1:]
        probabilities.sort()
        
        metrics_list = []
        for p in probabilities:
            y_pred = np.where(pred[:, idx] < p, 0, 1)
            metric = metric_func(y_true, y_pred)
            metrics_list.append(metric)
        
        if spline_k is not None: 
            try:
                spl = UnivariateSpline(probabilities, metrics_list, k=spline_k)
                spl_y = spl(probabilities)
                # get optimal thresholds on the spline and on the val_metric_list
                best_index = np.argmax(spl_y)
            except: 
                best_index = np.argmax(metrics_list)
        else:
            best_index = np.argmax(metrics_list)
        
        best_p = probabilities[best_index]
        best_metric = metrics_list[best_index]
        if verbose: 
            print("Best metric for {} is {}. threshold = {}.".format(cxr_label, best_metric, best_p))
        
        best_p_vals[cxr_label] = best_p
    return best_p_vals
   
def compute_f1_mcc(X_test_dir, y_test_dir, X_val_dir, y_val_dir, model_path, alt_labels_dict : dict = None, find_best_alt: bool = True, thresh_func: Callable = matthews_corrcoef): 
    """
    Computes f1 and mcc scores given test dataset, validation dataset (to find
    thresholds) and path to the model.
    
    Params: 
    * find_best_alt : bool
        If True, will filter alt_labels_dict to only the best alternative labels
        based on the validation dataset. Otherwise, will run on all alternative labels
        provided.
    """
    
    

    # specify basic cxr labels
    cxr_labels = ['Atelectasis','Cardiomegaly', 
                                      'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                      'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                                      'Pneumothorax', 'Support Devices']
    
    # load in ground truth
    VAL_GROUNDTRUTH_PATH = "val_groundtruth.csv"
    GROUNDTRUTH_PATH = "groundtruth.csv"
    
    val_groundtruth = y_true_csv_to_np(VAL_GROUNDTRUTH_PATH, cxr_labels)
    groundtruth = y_true_csv_to_np(GROUNDTRUTH_PATH, cxr_labels)

    NUM_LABELS = 14
     
    # run evaluation on validation and test datasets
    # dir for validation datasets
    val_X = "/deep/group/data/med-data/valid.h5"
    val_y = "/deep/group/data/CheXpert-320x320/valid.csv"
    
    # dir for test datasets
    test_X = "/deep/group/data/med-data/test_cxr.h5"
    test_y = "/deep/group/data/med-data/final_paths.csv"
    
    if alt_labels_dict is not None and find_best_alt: 
        # find best alternate labels
        val_res, val_pred = evaluate_model(val_X, val_y, best_model_path, alt_labels_dict=alt_labels_dict)
        # save alternative label results on validation dataset
        alt_val_res = val_res[0][1]
        best_alt_labels_dict = get_best_alt_labels(alt_val_res, cxr_labels)
    elif alt_labels_dict is not None: # find_best_alt == False
        best_alt_labels_dict = alt_labels_dict
    else: # no alternative labels
        best_alt_labels_dict = None
    
    # create alt_labels
    if best_alt_labels_dict is not None: 
        alt_labels_list, alt_label_idx_map = process_alt_labels(best_alt_labels_dict, cxr_labels)
    else: 
        alt_labels_list, alt_label_idx_map = cxr_labels, None
    
    # TODO: convert preds into binarized and make this one of the things that are returned
    val_res, val_pred = evaluate_model(val_X, val_y, model_path, cxr_labels, alt_labels_dict=best_alt_labels_dict)
    test_res, test_pred = evaluate_model(test_X, test_y, model_path, cxr_labels, alt_labels_dict=best_alt_labels_dict)
    
    # get best thresholds
    best_p_vals = get_best_p_vals(val_pred, val_groundtruth, alt_labels_list, alt_label_idx_map, metric_func=thresh_func)
    
    # f1 computation
    f1_cis = compute_f1(test_pred, groundtruth, alt_labels_list, best_p_vals, alt_label_idx_map)
    # mcc computation
    mcc_cis = compute_mcc(test_pred, groundtruth, alt_labels_list, best_p_vals, alt_label_idx_map)
    
    return f1_cis, mcc_cis    

def compute_f1(y_pred, y_true, cxr_labels, thresholds, label_idx_map=None): 
    def get_f1_clip_bootstrap(y_pred, y_true, best_p_vals, cxr_labels=cxr_labels, label_idx_map=None):
        stats = {}
        probs = np.copy(y_pred)
        for idx, cxr_label in enumerate(cxr_labels):
            p = best_p_vals[cxr_label]
            probs[:,idx] = np.where(probs[:,idx] < p, 0, 1)
        clip_preds = np.copy(probs)
        for idx, cxr_label in enumerate(cxr_labels):

            if label_idx_map is None: 
                curr_y_true = y_true[:, idx]
            else: 
                curr_y_true = y_true[:, label_idx_map[cxr_label]]
            curr_y_pred = clip_preds[:, idx]

            m = confusion_matrix(curr_y_true, curr_y_pred)
            if len(m.ravel()) == 1:
                tn = 500
                fp = 0
                fn = 0
                tp = 0
            else:
                tn, fp, fn, tp = m.ravel()

            if ((2*tp + fp +fn) == 0):
                stats[cxr_label] = 1
                continue

            stats[cxr_label] = [(2 * tp) / (2*tp + fp +fn)]
        # compute mean over five major pathologies
        stats["Mean"] = compute_mean(stats, is_df=False)
        return pd.DataFrame.from_dict(stats)

    boot_stats, f1_cis = f1_mcc_bootstrap(y_pred, y_true, cxr_labels, thresholds, get_f1_clip_bootstrap, n_samples=1000, label_idx_map=label_idx_map)
    return f1_cis
    
def compute_mcc(y_pred: np.array, y_true: np.array, cxr_labels: List, thresholds: dict, label_idx_map: dict = None): 
    def get_mcc_bootstrap(y_pred, y_true, best_p_vals, cxr_labels=cxr_labels, label_idx_map=None):
        stats = {}
        probs = np.copy(y_pred)

        for idx, cxr_label in enumerate(cxr_labels):
            p = best_p_vals[cxr_label]
            probs[:,idx] = np.where(probs[:,idx] < p, 0, 1)

        clip_preds = np.copy(probs)

        for idx, cxr_label in enumerate(cxr_labels):
            if label_idx_map is None: 
                curr_y_true = y_true[:, idx]
            else: 
                curr_y_true = y_true[:, label_idx_map[cxr_label]]

            curr_y_pred = clip_preds[:, idx]
            stats[cxr_label] = [matthews_corrcoef(curr_y_true, curr_y_pred)]
        # compute mean over five major pathologies
        stats["Mean"] = compute_mean(stats, is_df=False)
        return pd.DataFrame.from_dict(stats)
    
    boot_stats, mcc_cis = f1_mcc_bootstrap(y_pred, y_true, cxr_labels, thresholds, get_mcc_bootstrap, n_samples=1000, label_idx_map=label_idx_map)
    return mcc_cis
    