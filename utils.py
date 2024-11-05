"""
Utility functions for CSAI model training and evaluation.

This module provides helper functions for:
- Data preprocessing and normalization
- Evaluation metrics calculation
- Visualization
- Training/evaluation loops
- Miscellaneous helper functions 
"""
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve, 
    balanced_accuracy_score, recall_score, precision_score, f1_score, 
    PrecisionRecallDisplay, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay
    )
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
import copy
import os
from torch.optim.optimizer import Optimizer
import math
matplotlib.use('Agg')


def normalize_plt(array):
    return (array - array.min()) / (array.max() - array.min())

def get_polarfig(args, replacement_probabilities, missing_rates, feature_mae, dir, fold, phase):

    replacement_probabilities = normalize_plt(replacement_probabilities)
    missing_rates = normalize_plt(missing_rates)
    feature_mae = normalize_plt(feature_mae)
    
    angles = np.linspace(0, 2 * np.pi, len(args.attributes), endpoint=False).tolist()
    stats = np.concatenate((replacement_probabilities,[replacement_probabilities[0]]))
    stats2 = np.concatenate((missing_rates,[missing_rates[0]]))
    stats3 = np.concatenate((feature_mae,[feature_mae[0]]))
    angles += angles[:1]

    plt.figure(figsize=(12, 12))
    plt.polar(angles, stats, label='Replacement_probabilities')
    plt.polar(angles, stats2, label='Missing Rates')
    plt.polar(angles, stats3, label='Feature MAE')

    plt.fill(angles, stats, alpha=0.3)
    plt.fill(angles, stats2, alpha=0.3)
    plt.fill(angles, stats3, alpha=0.3)
    # plt.title('Radar Chart', position=(0.5, 1.1))
    plt.xticks(angles[:-1], args.attributes)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('{}/fold_{}_{}_best_{}.png'.format(dir, fold, args.task, phase), dpi=500)

# Define the function to print and write log
def writelog(file, line):
    file.write(line + '\n')
    print(line)

def normalize(data, mean, std, compute_intervals=False):
    n_patients = data.shape[0]
    n_hours = data.shape[1]
    n_variables = data.shape[2]

    measure = copy.deepcopy(data).reshape(n_patients * n_hours, n_variables)

    if compute_intervals:
        intervals_list = {v: [] for v in range(n_variables)}

    isnew = 0
    if len(mean) == 0 or len(std) == 0:
        isnew = 1
        mean_set = np.zeros([n_variables])
        std_set = np.zeros([n_variables])
    else:
        mean_set = mean
        std_set = std
        
    for v in range(n_variables):

        if isnew:
            mask_v = ~np.isnan(measure[:,v]) * 1
            idx_global = np.where(mask_v == 1)[0]

            if idx_global.sum() == 0:
                continue

            measure_mean = np.mean(measure[:, v][idx_global])
            measure_std = np.std(measure[:, v][idx_global])

            mean_set[v] = measure_mean
            std_set[v] = measure_std
        else:
            measure_mean = mean[v]
            measure_std = std[v]

        for p in range(n_patients):
            mask_p_v = ~np.isnan(data[p, :, v]) * 1
            idx_p = np.where(mask_p_v == 1)[0]

            if compute_intervals and len(idx_p) > 1:
                intervals_list[v].extend([idx_p[i+1] - idx_p[i] for i in range(len(idx_p)-1)])

            for ix in idx_p:
                if measure_std != 0:
                    data[p, ix, v] = (data[p, ix, v] - measure_mean) / measure_std
                else:
                    data[p, ix, v] = data[p, ix, v] - measure_mean

    if compute_intervals:
        intervals_list = {v: np.median(intervals_list[v]) if intervals_list[v] else np.nan for v in intervals_list}

    if compute_intervals:
        return data, mean_set, std_set, intervals_list
    else:
        return data, mean_set, std_set


def collate_fn_bidirectional(recs):

    def to_tensor_dict(recs):

        values = torch.FloatTensor(np.array([r['values'] for r in recs]))
        last_obs_f = torch.FloatTensor(np.array([r['last_obs_f'] for r in recs]))
        last_obs_b = torch.FloatTensor(np.array([r['last_obs_b'] for r in recs]))
        masks = torch.FloatTensor(np.array([r['masks'] for r in recs]))
        deltas_f = torch.FloatTensor(np.array([r['deltas_f'] for r in recs]))
        deltas_b = torch.FloatTensor(np.array([r['deltas_b'] for r in recs]))

        evals = torch.FloatTensor(np.array([r['evals'] for r in recs]))
        eval_masks = torch.FloatTensor(np.array([r['eval_masks'] for r in recs]))

        return {'values': values,
                'last_obs_f': last_obs_f,
                'last_obs_b': last_obs_b,
                'masks': masks,
                'deltas_f': deltas_f,
                'deltas_b': deltas_b,
                'evals': evals,
                'eval_masks': eval_masks}

    ret_dict = to_tensor_dict(recs)

    ret_dict['labels'] = torch.FloatTensor(np.array([r['label'] for r in recs]))

    return ret_dict

def parse_delta_bidirectional(masks, direction):
    if direction == 'backward':
        masks = masks[::-1] == 1.0

    [T, D] = masks.shape
    deltas = []
    for t in range(T):
        if t == 0:
            deltas.append(np.ones(D))
        else:
            deltas.append(np.ones(D) + (1 - masks[t]) * deltas[-1])

    return np.array(deltas)

def compute_last_obs(data, masks, direction):
    """
    Compute the last observed values for each time step.
    
    Parameters:
    - data (np.array): Original data array.
    - masks (np.array): Binary masks indicating where data is not NaN.

    Returns:
    - last_obs (np.array): Array of the last observed values.
    """
    if direction == 'backward':
        masks = masks[::-1] == 1.0
        data = data[::-1]
    
    [T, D] = masks.shape
    last_obs = np.full((T, D), np.nan)  # Initialize last observed values with NaNs
    last_obs_val = np.full(D, np.nan)  # Initialize last observed values for first time step with NaNs
    
    for t in range(1, T):
        mask = masks[t-1]
        # Update last observed values
        # last_obs_val = np.where(masks[t], data[t], last_obs_val) 
        last_obs_val[mask] = data[t-1, mask]
        # Assign last observed values to the current time step
        last_obs[t] = last_obs_val 
    
    return last_obs

def adjust_probability_vectorized(obs_count, avg_count, base_prob, increase_factor=0.5):
    if obs_count < avg_count:
        return min(base_prob * (avg_count / obs_count) * increase_factor, 1.0)
    else:
        return max(base_prob * (obs_count / avg_count) / increase_factor, 0)

def non_uniform_sample_loader_bidirectional(data, label, batch_size, removal_percent, pre_replacement_probabilities=None, increase_factor=0.5, times=1, shuffle=True):

    # Random seed
    np.random.seed(1)
    torch.manual_seed(1)

    # Get Dimensionality
    [N, T, D] = data.shape

    if pre_replacement_probabilities is None:

        observations_per_feature = np.sum(~np.isnan(data), axis=(0, 1))
        average_observations = np.mean(observations_per_feature)
        replacement_probabilities = np.full(D, removal_percent / 100)

        if increase_factor > 0:
            print('The increase_factor is {}!'.format(increase_factor))
            for feature_idx in range(D):
                replacement_probabilities[feature_idx] = adjust_probability_vectorized(
                    observations_per_feature[feature_idx],
                    average_observations,
                    replacement_probabilities[feature_idx],
                    increase_factor=increase_factor
                )
            
            # print('before:\n',replacement_probabilities)
            total_observations = np.sum(observations_per_feature)
            total_replacement_target = total_observations * removal_percent / 100

            for _ in range(1000):  # Limit iterations to prevent infinite loop
                total_replacement = np.sum(replacement_probabilities * observations_per_feature)
                if np.isclose(total_replacement, total_replacement_target, rtol=1e-3):
                    break
                adjustment_factor = total_replacement_target / total_replacement
                replacement_probabilities *= adjustment_factor
            
            # print('after:\n',replacement_probabilities)
    else:
        replacement_probabilities = pre_replacement_probabilities

    recs = []
    number = 0
    masks_sum = np.zeros(D)
    eval_masks_sum = np.zeros(D)
    values = copy.deepcopy(data)
    random_matrix = np.random.rand(N, T, D)
    values[(~np.isnan(values)) & (random_matrix < replacement_probabilities)] = np.nan
    for i in range(N):
        masks = ~np.isnan(values[i, :, :])
        eval_masks = (~np.isnan(values[i, :, :])) ^ (~np.isnan(data[i, :, :]))
        evals = data[i, :, :]
        rec = {}
        rec['label'] = label[i]
        deltas_f = parse_delta_bidirectional(masks, direction='forward')
        deltas_b = parse_delta_bidirectional(masks, direction='backward')
        last_obs_f = compute_last_obs(values[i, :, :], masks, direction='forward')
        last_obs_b = compute_last_obs(values[i, :, :], masks, direction='backward')
        rec['values'] = np.nan_to_num(values[i, :, :]).tolist()
        rec['last_obs_f'] = np.nan_to_num(last_obs_f).tolist()
        rec['last_obs_b'] = np.nan_to_num(last_obs_b).tolist()
        rec['masks'] = masks.astype('int32').tolist()
        rec['evals'] = np.nan_to_num(evals).tolist()
        rec['eval_masks'] = eval_masks.astype('int32').tolist()
        rec['deltas_f'] = deltas_f.tolist()
        rec['deltas_b'] = deltas_b.tolist()
        recs.append(rec)
        number += 1
        masks_sum += np.sum(masks, axis=0)
        eval_masks_sum += np.sum(eval_masks, axis=0)

    print('The number of valid sample is {}'.format(number))
    # Define the loader
    loader = DataLoader(recs,
                        batch_size=batch_size,
                        num_workers=1,
                        shuffle=shuffle,
                        pin_memory=True,
                        collate_fn=collate_fn_bidirectional)
    
    return loader, replacement_probabilities

def calculate_performance(y, y_score, y_pred, pr_display=False, cm_display=False, roc_display=False):
    # Calculate Evaluation Metrics
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_score)
    prec_macro = precision_score(y, y_pred, average='macro')
    recall_macro = recall_score(y, y_pred, average='macro')
    f1_macro = f1_score(y, y_pred, average='macro')
    bal_acc = balanced_accuracy_score(y, y_pred)

    if pr_display:
        prec, recall, _ = precision_recall_curve(y, y_score)
        PR_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

    if cm_display:
        cm = confusion_matrix(y, y_pred).ravel()
        CM_display = ConfusionMatrixDisplay(cm).plot()

    if roc_display:
        fpr, tpr, _ = roc_curve(y, y_score).ravel()
        ROC_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

    return acc, auc, prec_macro, recall_macro, f1_macro, bal_acc

def setup_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def training(phase, model, optimizer, criterion, data, args, task, f, tfw, epoch):
    # Set mode as Training
    model.train()

    loss = 0
    loss_imputation = 0
    loss_classification = 0
    eval_x_all = []
    eval_m_all = []
    imp_all = []
    y_gts = np.array([]).reshape(0)
    y_preds = np.array([]).reshape(0)
    y_scores = np.array([]).reshape(0)

    # Loop over the minibatch
    for i, xdata in enumerate(data):
        # Define training variables
        y = xdata['labels'].to(args.device)
        eval_x = xdata['evals'].to(args.device)
        eval_m = xdata['eval_masks'].to(args.device)
        
        # Zero Grad
        optimizer.zero_grad()
        
        ret = model(xdata)

        loss_imputation += (ret['loss_regression'] + ret['loss_consistency']).item()

        # Loss
        if task == 'C':
            BCE_f, _ = criterion(ret['y_score_f'], ret['y_out_f'], y.unsqueeze(1))
            BCE_b, _ = criterion(ret['y_score_b'], ret['y_out_b'], y.unsqueeze(1))
            loss_classification += (BCE_f + BCE_b).item()
            loss_total = args.imputation_weight * ret['loss_regression'] + args.classification_weight * (BCE_f + BCE_b) + args.consistency_weight * ret['loss_consistency']
        else:
            # Overall loss
            loss_total = args.imputation_weight * ret['loss_regression'] + args.consistency_weight * ret['loss_consistency']

        loss += loss_total.item()
        # Bacward Propagation and Update the weights
        loss_total.backward()
        # Update the weights
        optimizer.step()
        
        # Calculate Evaluation Metric
        eval_m = eval_m.data.cpu().numpy()
        ex = eval_x.data.cpu().numpy()
        impx = ret['imputation'].data.cpu().numpy()
        eval_x_all.append(ex)
        eval_m_all.append(eval_m)
        imp_all.append(impx)

        if task == 'C':
            y_gts = np.hstack([y_gts, y.to('cpu').detach().numpy().flatten()])
            y_score = (ret['y_score_f'] + ret['y_score_b']) / 2
            y_score = y_score.to('cpu').detach().numpy()
            y_out = np.round(y_score)
            y_preds = np.hstack([y_preds, y_out.reshape(-1)])
            y_scores = np.hstack([y_scores, y_score.reshape(-1)])

    # Averaging the loss
    loss = loss / (i+1)
    loss_imputation = loss_imputation / (i+1)
    loss_classification = loss_classification / (i+1)
    eval_x_all = np.concatenate(eval_x_all, axis=0, dtype=np.float64)
    eval_m_all = np.concatenate(eval_m_all, axis=0, dtype=np.float64)
    imp_all = np.concatenate(imp_all, axis=0, dtype=np.float64)

    writelog(f, 'Loss : ' + str(loss))
    writelog(f, 'Loss_imputation : ' + str(loss_imputation))
    writelog(f, 'Loss_classification : ' + str(loss_classification))

    mae = (np.abs(eval_x_all - imp_all)*eval_m_all).sum()/eval_m_all.sum()
    mre = (np.abs(eval_x_all - imp_all)*eval_m_all).sum() / (np.abs(eval_x_all)*eval_m_all).sum()
    feature_mae = (np.abs(eval_x_all - imp_all)*eval_m_all).mean((0,1))

    # Averaging & displaying the Evaluation Metric
    writelog(f, 'MAE : ' + str(mae))
    writelog(f, 'MRE : ' + str(mre))

    # Tensorboard Logging
    results = {'loss': loss,
            'loss_imputation': loss_imputation,
            'mae': mae,
            'mre': mre,
            }
                        
    if task == 'C':
        acc, auc, prec_macro, recall_macro, f1_macro, bal_acc = calculate_performance(y_gts, y_scores, y_preds)

        # Averaging & displaying the Evaluation Metric      
        writelog(f, 'Accuracy : ' + str(acc))
        writelog(f, 'AUC : ' + str(auc))
        writelog(f, 'Precision macro : ' + str(prec_macro))
        writelog(f, 'Recall macro : ' + str(recall_macro))
        writelog(f, 'F1 macro : ' + str(f1_macro))
        writelog(f, 'Balanced accuracy : ' + str(bal_acc))

        # Tensorboard Logging
        results2 = {'loss_classification': loss_classification,
                 'accuracy': acc,
                 'auc': auc,
                 'prec_macro': prec_macro,
                 'recall_macro': recall_macro,
                 'f1_macro': f1_macro,
                 'bal_acc': bal_acc
                }
        results.update(results2)

    for tag, value in results.items():
        tfw[phase].add_scalar(tag=tag, scalar_value=value, global_step=epoch)


    results.update({'feature_mae': feature_mae})
    return results

def evaluate(phase, model, criterion, data, args, task, f, tfw, epoch):
    # Set mode as Evaluation
    model.eval()

    loss = 0
    loss_classification = 0
    loss_imputation = 0
    eval_x_all = []
    eval_m_all = []
    imp_all = []
    y_gts = np.array([]).reshape(0)
    y_preds = np.array([]).reshape(0)
    y_scores = np.array([]).reshape(0)

    # Loop over the minibatch
    with torch.no_grad():
        for i, xdata in enumerate(data):
            # xdata = next(iter(data))
            y = xdata['labels'].to(args.device)
            eval_x = xdata['evals'].to(args.device)
            eval_m = xdata['eval_masks'].to(args.device)

            ret = model(xdata)

            loss_imputation += (ret['loss_regression'] + ret['loss_consistency']).item()

            # Loss
            if task in ['C', 'pretrain', 'pretrain_brits', 'pretrain_train',]:
                BCE_f, _ = criterion(ret['y_score_f'], ret['y_out_f'], y.unsqueeze(1))
                BCE_b, _ = criterion(ret['y_score_b'], ret['y_out_b'], y.unsqueeze(1))
                loss_classification += (BCE_f + BCE_b).item()
                loss_total = args.imputation_weight * ret['loss_regression'] + args.classification_weight * (BCE_f + BCE_b) + args.consistency_weight * ret['loss_consistency']
            else:
                # Overall loss
                loss_total = args.imputation_weight * ret['loss_regression'] + args.consistency_weight * ret['loss_consistency']

            loss += loss_total.item()

            # Calculate Evaluation Metric
            eval_m = eval_m.data.cpu().numpy()
            ex = eval_x.data.cpu().numpy()
            impx = ret['imputation'].data.cpu().numpy()
            eval_x_all.append(ex)
            eval_m_all.append(eval_m)
            imp_all.append(impx)
            
            if task == 'C':
                y_gts = np.hstack([y_gts, y.to('cpu').detach().numpy().flatten()])
                y_score = (ret['y_score_f'] + ret['y_score_b']) / 2
                y_score = y_score.to('cpu').detach().numpy()
                y_out = np.round(y_score)
                y_preds = np.hstack([y_preds, y_out.reshape(-1)])
                y_scores = np.hstack([y_scores, y_score.reshape(-1)])

    # Averaging the loss
    loss = loss / (i+1)
    loss_imputation = loss_imputation / (i+1)
    loss_classification = loss_classification / (i+1)
    eval_x_all = np.concatenate(eval_x_all, axis=0, dtype=np.float64)
    eval_m_all = np.concatenate(eval_m_all, axis=0, dtype=np.float64)
    imp_all = np.concatenate(imp_all, axis=0, dtype=np.float64)

    writelog(f, 'Loss : ' + str(loss))
    writelog(f, 'Loss imputation : ' + str(loss_imputation))
    writelog(f, 'Loss classification : ' + str(loss_classification))

    mae = (np.abs(eval_x_all - imp_all)*eval_m_all).sum()/eval_m_all.sum()
    mre = (np.abs(eval_x_all - imp_all)*eval_m_all).sum() / (np.abs(eval_x_all)*eval_m_all).sum()
    feature_mae = (np.abs(eval_x_all - imp_all)*eval_m_all).mean((0,1))

    # Averaging & displaying the Evaluation Metric
    writelog(f, 'MAE : ' + str(mae))
    writelog(f, 'MRE : ' + str(mre))

    # Tensorboard Logging
    results = {'loss': loss,
            'loss_imputation': loss_imputation,
            'mae': mae,
            'mre': mre,
            }

    if task == 'C':
        acc, auc, prec_macro, recall_macro, f1_macro, bal_acc = calculate_performance(y_gts, y_scores, y_preds)

        # Averaging & displaying the Evaluation Metric      
        writelog(f, 'Accuracy : ' + str(acc))
        writelog(f, 'AUC : ' + str(auc))
        writelog(f, 'Precision macro : ' + str(prec_macro))
        writelog(f, 'Recall macro : ' + str(recall_macro))
        writelog(f, 'F1 macro : ' + str(f1_macro))
        writelog(f, 'Balanced accuracy : ' + str(bal_acc))

        # Tensorboard Logging
        results2 = {'loss_classification': loss_classification,
                 'accuracy': acc,
                 'auc': auc,
                 'prec_macro': prec_macro,
                 'recall_macro': recall_macro,
                 'f1_macro': f1_macro,
                 'bal_acc': bal_acc
                }
        
        results.update(results2)

    for tag, value in results.items():
        tfw[phase].add_scalar(tag=tag, scalar_value=value, global_step=epoch)

    results.update({'feature_mae': feature_mae})
    return results

# Configuration
def config(args):
    args.data_path = './data/' + args.dataset + '/data_nan.pkl'
    args.label_path = './data/' + args.dataset + '/label.pkl'

    setup_seed(args.seed)

    if args.dataset == 'mimic_59f':
        args.vae_hiddens = [59, 128, 32, 16]
        args.attributes = ['Capillary refill rate-0', 'Capillary refill rate-1', 'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glascow coma scale eye opening-To Pain', 'Glascow coma scale eye opening-3 To speech',
            'Glascow coma scale eye opening-1 No Response', 'Glascow coma scale eye opening-4 Spontaneously', 'Glascow coma scale eye opening-None', 'Glascow coma scale eye opening-To Speech', 'Glascow coma scale eye opening-Spontaneously',
            'Glascow coma scale eye opening-2 To pain', 'Glascow coma scale motor response-1 No Response', 'Glascow coma scale motor response-3 Abnorm flexion', 'Glascow coma scale motor response-Abnormal extension',
            'Glascow coma scale motor response-No response', 'Glascow coma scale motor response-4 Flex-withdraws', 'Glascow coma scale motor response-Localizes Pain', 'Glascow coma scale motor response-Flex-withdraws',
            'Glascow coma scale motor response-Obeys Commands', 'Glascow coma scale motor response-Abnormal Flexion', 'Glascow coma scale motor response-6 Obeys Commands', 'Glascow coma scale motor response-5 Localizes Pain',
            'Glascow coma scale motor response-2 Abnorm extensn', 'Glascow coma scale total-11', 'Glascow coma scale total-10', 'Glascow coma scale total-13', 'Glascow coma scale total-12', 'Glascow coma scale total-15',
            'Glascow coma scale total-14', 'Glascow coma scale total-3', 'Glascow coma scale total-5', 'Glascow coma scale total-4', 'Glascow coma scale total-7', 'Glascow coma scale total-6', 'Glascow coma scale total-9',
            'Glascow coma scale total-8', 'Glascow coma scale verbal response-1 No Response', 'Glascow coma scale verbal response-No Response', 'Glascow coma scale verbal response-Confused', 'Glascow coma scale verbal response-Inappropriate Words',
            'Glascow coma scale verbal response-Oriented', 'Glascow coma scale verbal response-No Response-ETT', 'Glascow coma scale verbal response-5 Oriented', 'Glascow coma scale verbal response-Incomprehensible sounds',
            'Glascow coma scale verbal response-1.0 ET/Trach', 'Glascow coma scale verbal response-4 Confused', 'Glascow coma scale verbal response-2 Incomp sounds', 'Glascow coma scale verbal response-3 Inapprop words', 'Glucose',
            'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']  

    elif args.dataset == 'physionet':
        args.vae_hiddens = [35, 64, 24, 10]
        args.attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2', 'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
            'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP', 'Creatinine', 'ALP']

    elif args.dataset == 'eicu':
        args.vae_hiddens = [20, 128, 32, 16]
        args.attributes = ['apacheadmissiondx', 'ethnicity', 'gender', 'GCS Total', 'Eyes', 'Motor', 'Verbal', 'admissionheight', 'admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)', 'Invasive BP Diastolic', 'Invasive BP Systolic',
            'O2 Saturation', 'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH']

    return args