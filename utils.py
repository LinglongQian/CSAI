"""
Utility functions for CSAI model training and evaluation.

This module contains helper functions for:
- Data normalization and preprocessing
- Metrics calculation and visualization
- Training/evaluation loops
- Logging utilities
"""
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve, 
    balanced_accuracy_score, recall_score, precision_score, f1_score, 
    PrecisionRecallDisplay, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay
)
import logging
import copy
import pickle

from torch.utils.tensorboard import SummaryWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================ Data Processing Functions ================
def normalize(data: np.ndarray, 
             mean: List[float] = [], 
             std: List[float] = [], 
             compute_intervals: bool = False
            ) -> Union[Tuple[np.ndarray, List[float], List[float]], 
                      Tuple[np.ndarray, List[float], List[float], Dict[int, float]]]:
    """
    Normalize time series data.
    
    Args:
        data: Input data array of shape [n_patients, n_timesteps, n_features]
        mean: Optional pre-computed mean values
        std: Optional pre-computed standard deviation values
        compute_intervals: Whether to compute feature intervals
        
    Returns:
        Tuple of:
        - Normalized data
        - Mean values
        - Standard deviation values
        - Feature intervals (if compute_intervals=True)
    """
    n_patients, n_hours, n_variables = data.shape
    measure = copy.deepcopy(data).reshape(n_patients * n_hours, n_variables)
    
    if compute_intervals:
        intervals_list = {v: [] for v in range(n_variables)}
    
    # Initialize or use provided statistics
    if not mean or not std:
        mean_set = np.zeros([n_variables])
        std_set = np.zeros([n_variables])
        compute_stats = True
    else:
        mean_set = mean
        std_set = std
        compute_stats = False
        
    # Process each variable
    for v in range(n_variables):
        if compute_stats:
            # Compute statistics excluding missing values
            mask_v = ~np.isnan(measure[:,v])
            if mask_v.sum() == 0:
                continue
                
            measure_mean = np.mean(measure[mask_v, v])
            measure_std = np.std(measure[mask_v, v])
            
            mean_set[v] = measure_mean
            std_set[v] = measure_std
        else:
            measure_mean = mean[v]
            measure_std = std[v]
            
        # Normalize each patient's data
        for p in range(n_patients):
            mask_p_v = ~np.isnan(data[p, :, v])
            idx_p = np.where(mask_p_v)[0]
            
            if compute_intervals and len(idx_p) > 1:
                intervals_list[v].extend([idx_p[i+1] - idx_p[i] for i in range(len(idx_p)-1)])
                
            for ix in idx_p:
                if measure_std != 0:
                    data[p, ix, v] = (data[p, ix, v] - measure_mean) / measure_std
                else:
                    data[p, ix, v] = data[p, ix, v] - measure_mean
                    
    if compute_intervals:
        intervals_list = {v: np.median(intervals_list[v]) if intervals_list[v] else np.nan 
                         for v in intervals_list}
        return data, mean_set, std_set, intervals_list
        
    return data, mean_set, std_set

def parse_delta_bidirectional(masks: np.ndarray, direction: str) -> np.ndarray:
    """
    Parse delta values for bidirectional model.
    
    Args:
        masks: Binary mask array
        direction: 'forward' or 'backward'
        
    Returns:
        Array of delta values
    """
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

def compute_last_obs(data: np.ndarray, masks: np.ndarray, direction: str) -> np.ndarray:
    """
    Compute last observed values at each timestep.
    
    Args:
        data: Input data array
        masks: Binary mask array
        direction: 'forward' or 'backward'
        
    Returns:
        Array of last observed values
    """
    if direction == 'backward':
        masks = masks[::-1] == 1.0
        data = data[::-1]
    
    [T, D] = masks.shape
    last_obs = np.full((T, D), np.nan)
    last_obs_val = np.full(D, np.nan)
    
    for t in range(1, T):
        mask = masks[t-1]
        last_obs_val[mask] = data[t-1, mask]
        last_obs[t] = last_obs_val
    
    return last_obs

# ================ Data Loading Functions ================

def collate_fn_bidirectional(recs: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for bidirectional data loading.
    
    Args:
        recs: List of record dictionaries containing time series data
        
    Returns:
        Dictionary of batched tensors
    """
    def to_tensor_dict(recs):
        values = torch.FloatTensor(np.array([r['values'] for r in recs]))
        last_obs_f = torch.FloatTensor(np.array([r['last_obs_f'] for r in recs]))
        last_obs_b = torch.FloatTensor(np.array([r['last_obs_b'] for r in recs]))
        masks = torch.FloatTensor(np.array([r['masks'] for r in recs]))
        deltas_f = torch.FloatTensor(np.array([r['deltas_f'] for r in recs]))
        deltas_b = torch.FloatTensor(np.array([r['deltas_b'] for r in recs]))
        evals = torch.FloatTensor(np.array([r['evals'] for r in recs]))
        eval_masks = torch.FloatTensor(np.array([r['eval_masks'] for r in recs]))

        return {
            'values': values,
            'last_obs_f': last_obs_f,
            'last_obs_b': last_obs_b,
            'masks': masks,
            'deltas_f': deltas_f,
            'deltas_b': deltas_b,
            'evals': evals,
            'eval_masks': eval_masks
        }

    ret_dict = to_tensor_dict(recs)
    ret_dict['labels'] = torch.FloatTensor(np.array([r['label'] for r in recs]))
    return ret_dict

def adjust_probability_vectorized(obs_count: float, 
                                avg_count: float, 
                                base_prob: float,
                                increase_factor: float = 0.5) -> float:
    """
    Adjust probability based on observation counts.
    
    Args:
        obs_count: Number of observations
        avg_count: Average number of observations
        base_prob: Base probability
        increase_factor: Factor to adjust probability
        
    Returns:
        Adjusted probability
    """
    if obs_count < avg_count:
        return min(base_prob * (avg_count / obs_count) * increase_factor, 1.0)
    else:
        return max(base_prob * (obs_count / avg_count) / increase_factor, 0)

def non_uniform_sample_loader_bidirectional(
    data: np.ndarray,
    label: np.ndarray,
    batch_size: int,
    removal_percent: float,
    pre_replacement_probabilities: Optional[np.ndarray] = None,
    increase_factor: float = 0.5,
    shuffle: bool = True
) -> Tuple[DataLoader, np.ndarray]:
    """
    Create data loader with non-uniform sampling.
    
    Args:
        data: Input data array
        label: Labels array
        batch_size: Batch size
        removal_percent: Percentage of values to remove
        pre_replacement_probabilities: Optional pre-computed probabilities
        increase_factor: Factor for probability adjustment
        shuffle: Whether to shuffle data
        
    Returns:
        Tuple of (DataLoader, replacement_probabilities)
    """
    np.random.seed(1)
    torch.manual_seed(1)

    [N, T, D] = data.shape

    if pre_replacement_probabilities is None:
        # Calculate observation counts and average
        observations = np.sum(~np.isnan(data), axis=(0, 1))
        avg_observations = np.mean(observations)
        replacement_probs = np.full(D, removal_percent / 100)

        if increase_factor > 0:
            logger.info(f'Using increase factor: {increase_factor}')
            
            # Adjust probabilities based on observation counts
            for feat_idx in range(D):
                replacement_probs[feat_idx] = adjust_probability_vectorized(
                    observations[feat_idx],
                    avg_observations,
                    replacement_probs[feat_idx],
                    increase_factor
                )
            
            # Normalize probabilities to maintain overall removal percentage
            total_obs = np.sum(observations)
            target_replacements = total_obs * removal_percent / 100
            
            for _ in range(1000):
                total_replacements = np.sum(replacement_probs * observations)
                if np.isclose(total_replacements, target_replacements, rtol=1e-3):
                    break
                adjustment = target_replacements / total_replacements
                replacement_probs *= adjustment
    else:
        replacement_probs = pre_replacement_probabilities

    # Create records with masked values
    recs = []
    values = copy.deepcopy(data)
    random_matrix = np.random.rand(N, T, D)
    values[(~np.isnan(values)) & (random_matrix < replacement_probs)] = np.nan
    
    for i in range(N):
        masks = ~np.isnan(values[i])
        eval_masks = (~np.isnan(values[i])) ^ (~np.isnan(data[i]))
        evals = data[i]
        
        deltas_f = parse_delta_bidirectional(masks, 'forward')
        deltas_b = parse_delta_bidirectional(masks, 'backward')
        last_obs_f = compute_last_obs(values[i], masks, 'forward')
        last_obs_b = compute_last_obs(values[i], masks, 'backward')
        
        rec = {
            'values': np.nan_to_num(values[i]),
            'last_obs_f': np.nan_to_num(last_obs_f),
            'last_obs_b': np.nan_to_num(last_obs_b),
            'masks': masks.astype('int32'),
            'deltas_f': deltas_f,
            'deltas_b': deltas_b,
            'evals': np.nan_to_num(evals),
            'eval_masks': eval_masks.astype('int32'),
            'label': label[i]
        }
        recs.append(rec)

    logger.info(f'Created {len(recs)} valid samples')

    # Create data loader
    loader = DataLoader(
        recs,
        batch_size=batch_size,
        num_workers=1,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn_bidirectional
    )

    return loader, replacement_probs

# ================ Metrics and Evaluation Functions ================













def setup_experiment_tracking(args) -> Dict[str, str]:
    """
    Set up experiment directories and logging.
    
    Args:
        args: Configuration arguments
        
    Returns:
        Dictionary containing experiment paths
    """
    model_detail = f'{args.model_name}_remove_{args.removal_percent}'
    exp_root = Path(args.model_path) / args.dataset / model_detail / f'task_{args.task}'
    
    # Create experiment directories
    paths = {}
    for fold in range(args.n_folds):
        fold_dir = exp_root / f'fold_{fold}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (fold_dir / 'tflog').mkdir(exist_ok=True)
        (fold_dir / 'model_state').mkdir(exist_ok=True)
        
        paths[f'fold_{fold}'] = {
            'root': fold_dir,
            'log': fold_dir / 'log.txt',
            'tflog': fold_dir / 'tflog',
            'model_state': fold_dir / 'model_state'
        }
    
    return paths



def calculate_metrics(y_true: np.ndarray, 
                      y_score: np.ndarray, 
                      y_pred: np.ndarray,
                      pr_display: bool = False,
                      cm_display: bool = False,
                      roc_display: bool = False
                    ) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_score: Prediction scores
        y_pred: Binary predictions
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_score),
        'prec_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
    }
    
    # Create displays if requested
    if pr_display:
        # Precision-Recall curve
        prec, recall, _ = precision_recall_curve(y_true, y_score)
        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
        PrecisionRecallDisplay(precision=prec, recall=recall).plot(ax=ax_pr)
        ax_pr.set_title('Precision-Recall Curve')
        metrics['pr_curve'] = fig_pr
        
    if cm_display:
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax_cm)
        ax_cm.set_title('Confusion Matrix')
        metrics['confusion_matrix'] = fig_cm
        
    if roc_display:
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=metrics['auc']).plot(ax=ax_roc)
        ax_roc.set_title('ROC Curve')
        metrics['roc_curve'] = fig_roc
    
    # Return metrics dictionary if no displays requested,
    # otherwise return tuple for backward compatibility
    if any([pr_display, cm_display, roc_display]):
        return (
            metrics['accuracy'],
            metrics['auc'],
            metrics['prec_macro'],
            metrics['recall_macro'],
            metrics['f1_macro'],
            metrics['bal_acc']
        )
    
    return metrics

def _calculate_final_metrics(
    metrics: Dict[str, float],
    n_batches: int,
    eval_x_all: List[np.ndarray],
    eval_m_all: List[np.ndarray],
    imp_all: List[np.ndarray],
    y_true: List[float],
    y_score: List[float],
    y_pred: List[float],
    task: str
) -> Dict[str, float]:
    """
    Calculate final metrics after processing all batches.
    
    Args:
        metrics: Dictionary of accumulated metrics
        n_batches: Number of batches processed
        eval_x_all: List of evaluation data arrays
        eval_m_all: List of evaluation mask arrays
        imp_all: List of imputed data arrays
        y_true: List of true labels
        y_score: List of prediction scores
        y_pred: List of binary predictions
        task: Task type ('I' or 'C')
        
    Returns:
        Dictionary of final metrics
    """
    # Average accumulated losses
    for key in ['loss', 'loss_imputation', 'loss_classification']:
        if key in metrics:
            metrics[key] /= n_batches

    # Concatenate all arrays
    eval_x = np.concatenate(eval_x_all, axis=0)
    eval_m = np.concatenate(eval_m_all, axis=0)
    imp = np.concatenate(imp_all, axis=0)

    # Calculate imputation metrics
    # MAE (Mean Absolute Error)
    metrics['mae'] = np.sum(np.abs(eval_x - imp) * eval_m) / np.sum(eval_m)
    
    # MRE (Mean Relative Error)
    with np.errstate(divide='ignore', invalid='ignore'):
        metrics['mre'] = np.sum(np.abs(eval_x - imp) * eval_m) / \
                        np.sum(np.abs(eval_x) * eval_m)
    
    # Feature-wise MAE
    metrics['feature_mae'] = np.mean(np.abs(eval_x - imp) * eval_m, axis=(0, 1))

    # Calculate classification metrics if applicable
    if task == 'C' and y_true:
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        y_pred = np.array(y_pred)
        
        metrics.update({
            'accuracy': accuracy_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_score),
            'prec_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'bal_acc': balanced_accuracy_score(y_true, y_pred)
        })

    return metrics

# ================ Visualization Functions ================

def get_polarfig(args, 
                replacement_probabilities: np.ndarray,
                missing_rates: np.ndarray,
                feature_mae: np.ndarray,
                save_dir: Path,
                fold: int,
                phase: str):
    """
    Create polar plot of feature statistics.
    
    Args:
        args: Configuration arguments
        replacement_probabilities: Feature replacement probabilities
        missing_rates: Feature missing rates
        feature_mae: Feature-wise MAE values
        save_dir: Directory to save plot
        fold: Fold number
        phase: Training phase
    """
    # Normalize values for visualization
    replacement_probabilities = (replacement_probabilities - replacement_probabilities.min()) / \
                              (replacement_probabilities.max() - replacement_probabilities.min())
    missing_rates = (missing_rates - missing_rates.min()) / \
                   (missing_rates.max() - missing_rates.min())
    feature_mae = (feature_mae - feature_mae.min()) / \
                 (feature_mae.max() - feature_mae.min())
    
    # Setup polar plot
    angles = np.linspace(0, 2 * np.pi, len(args.attributes), endpoint=False)
    
    stats = np.concatenate((replacement_probabilities, [replacement_probabilities[0]]))
    stats2 = np.concatenate((missing_rates, [missing_rates[0]]))
    stats3 = np.concatenate((feature_mae, [feature_mae[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    # Create plot
    plt.figure(figsize=(12, 12))
    plt.polar(angles, stats, label='Replacement Probabilities')
    plt.polar(angles, stats2, label='Missing Rates')
    plt.polar(angles, stats3, label='Feature MAE')
    
    plt.fill(angles, stats, alpha=0.3)
    plt.fill(angles, stats2, alpha=0.3)
    plt.fill(angles, stats3, alpha=0.3)
    
    plt.xticks(angles[:-1], args.attributes)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_dir / f'fold_{fold}_{args.task}_best_{phase}.png', dpi=500)
    plt.close()

# ================ Configuration and Setup Functions ================

def config(args):
    """Configure dataset-specific parameters."""
    # Set data paths
    args.data_path = f'./data/{args.dataset}/data_nan.pkl'
    args.label_path = f'./data/{args.dataset}/label.pkl'

    setup_seed(args.seed)

    # Configure dataset-specific parameters
    dataset_configs = {
        'mimic_59f': {
            'vae_hiddens': [59, 128, 32, 16],
            'attributes': ['Capillary refill rate-0', 'Capillary refill rate-1', 'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glascow coma scale eye opening-To Pain', 'Glascow coma scale eye opening-3 To speech',
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
        },
        'physionet': {
            'vae_hiddens': [35, 64, 24, 10],
            'attributes': ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2', 'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
            'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP', 'Creatinine', 'ALP']
        },
        'eicu': {
            'vae_hiddens': [20, 128, 32, 16],
            'attributes': ['apacheadmissiondx', 'ethnicity', 'gender', 'GCS Total', 'Eyes', 'Motor', 'Verbal', 'admissionheight', 'admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)', 'Invasive BP Diastolic', 'Invasive BP Systolic',
            'O2 Saturation', 'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH']
        }
    }

    if args.dataset in dataset_configs:
        for key, value in dataset_configs[args.dataset].items():
            setattr(args, key, value)

    return args

def setup_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# ================ Logging and Experiment Tracking ================

class ExperimentLogger:
    """Helper class for experiment logging."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        
    def log(self, message: str):
        """Write message to log file and print to console."""
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
        logger.info(message)
        
    def log_config(self, args):
        """Log experiment configuration."""
        self.log('---------------')
        self.log(f'Model: {args.model_name}_remove_{args.removal_percent}')
        self.log(f'Hidden Units: {args.hiddens}')
        self.log(f'Times: {args.times}')
        self.log(f'Increase_factor: {args.increase_factor}')
        self.log(f'Step channels: {args.step_channels}')
        self.log(f'Task: {args.task}')
        self.log(f'Pre model: {args.pre_model}')
        self.log('---------------')
        self.log(f'Dataset: {args.dataset}')
        self.log(f'Hours: {args.hours}')
        self.log(f'Removal: {args.removal_percent}')
        self.log('---------------')
        self.log(f'Learning Rate: {args.lr:.5f}')
        self.log(f'Batch Size: {args.batchsize}')
        self.log(f'Weight decay: {args.weight_decay:.5f}')
        self.log(f'Imputation Weight: {args.imputation_weight:.3f}')
        self.log(f'Consistency Loss Weight: {args.consistency_weight:.3f}')
        self.log(f'Classification Weight: {args.classification_weight:.3f}')
        self.log('---------------')
        
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ''):
        """Log evaluation metrics."""
        for name, value in metrics.items():
            self.log(f'{prefix}{name}: {value:.6f}')

def save_training_info(
    info: Dict,
    save_dir: Path,
    fold: int,
    args
) -> None:
    """Save training information and create plots."""
    # Save training records
    with open(save_dir / f'training_recording_{fold}.pkl', 'wb') as f:
        pickle.dump(info, f, -1)
    
    # Create performance plots based on task
    if args.task == 'I':
        _plot_imputation_performance(info, save_dir, fold, args.task)
    else:
        _plot_imputation_performance(info, save_dir, fold, args.task)
        _plot_classification_performance(info, save_dir, fold, args.task)

def _plot_imputation_performance(
    info: Dict,
    save_dir: Path,
    fold: int,
    task: str
) -> None:
    """Create plots for imputation performance metrics."""
    fig, axes = plt.subplots(3, 1, figsize=(20, 20))
    
    # Plot overall losses
    axes[0].plot(info['train']['Loss'], label='Training')
    axes[0].plot(info['valid']['Loss'], label='Validation')
    axes[0].plot(info['test']['Loss'], label='Testing')
    axes[0].legend()
    axes[0].set_title('Overall losses over epochs')
    
    # Plot imputation losses
    axes[1].plot(info['train']['Loss_imputation'], label='Training')
    axes[1].plot(info['valid']['Loss_imputation'], label='Validation')
    axes[1].plot(info['test']['Loss_imputation'], label='Testing')
    axes[1].legend()
    axes[1].set_title('Losses of imputation over epochs')
    
    # Plot MAE
    axes[2].plot(info['train']['Mae'], label='Training')
    axes[2].plot(info['valid']['Mae'], label='Validation')
    axes[2].plot(info['test']['Mae'], label='Testing')
    axes[2].legend()
    axes[2].set_title('MAEs over epochs')
    
    plt.savefig(save_dir / f"fold_{fold}_{task}_performances.png", dpi=500)
    plt.close()

def _plot_classification_performance(
    info: Dict,
    save_dir: Path,
    fold: int,
    task: str
) -> None:
    """Create plots for classification performance metrics."""
    # Plot metrics over time
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # Performance metrics
    metrics = ['Auc', 'prec_macro', 'recall_macro', 'f1_macro', 'bal_acc']
    
    # Plot training metrics
    for metric in metrics:
        axes[0, 0].plot(info['train'][metric], label=metric)
    axes[0, 0].legend()
    axes[0, 0].set_title('Training performance over epochs')
    
    # Plot validation metrics
    for metric in metrics:
        axes[0, 1].plot(info['valid'][metric], label=metric)
    axes[0, 1].legend()
    axes[0, 1].set_title('Validation performance over epochs')
    
    # Plot test metrics
    for metric in metrics:
        axes[1, 0].plot(info['test'][metric], label=metric)
    axes[1, 0].legend()
    axes[1, 0].set_title('Test performance over epochs')
    
    # Plot AUC comparison
    axes[1, 1].plot(info['train']['Auc'], label='Training')
    axes[1, 1].plot(info['valid']['Auc'], label='Validation')
    axes[1, 1].plot(info['test']['Auc'], label='Testing')
    axes[1, 1].legend()
    axes[1, 1].set_title('AUCs over epochs')
    
    plt.savefig(save_dir / f"fold_{fold}_{task}_classification_performances.png", dpi=500)
    plt.close()

def is_better_metric(
    current: float,
    best: float,
    metric: str
) -> bool:
    """
    Determine if current metric is better than best metric.
    
    Args:
        current: Current metric value
        best: Best metric value so far
        metric: Name of metric
        
    Returns:
        True if current metric is better than best
    """
    higher_better = {
        'auc', 'accuracy', 'prec_macro', 'recall_macro', 
        'f1_macro', 'bal_acc'
    }
    lower_better = {
        'mae', 'mre', 'loss', 'loss_imputation', 
        'loss_classification'
    }
    
    if metric in higher_better:
        return current > best
    elif metric in lower_better:
        return current < best
    else:
        raise ValueError(f"Unknown metric for comparison: {metric}")
    
def evaluate(
    phase: str,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data: torch.utils.data.DataLoader,
    args,
    task: str,
    logger: ExperimentLogger,
    tfw: SummaryWriter,
    epoch: int
) -> Dict[str, float]:
    """
    Evaluate model on data.
    
    Args:
        phase: Evaluation phase ('valid' or 'test')
        model: Neural network model
        criterion: Loss function
        data: DataLoader for evaluation
        args: Configuration arguments
        task: Task type ('I' for imputation or 'C' for classification)
        logger: Logger instance
        tfw: TensorBoard writer
        epoch: Current epoch number
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    loss = 0
    loss_imputation = 0
    loss_classification = 0
    eval_x_all = []
    eval_m_all = []
    imp_all = []
    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for i, batch in enumerate(data):
            # Get batch data
            y = batch['labels'].to(args.device)
            eval_x = batch['evals'].to(args.device)
            eval_m = batch['eval_masks'].to(args.device)

            # Forward pass
            outputs = model(batch)
            
            # Calculate losses
            imp_loss = outputs['loss_regression'] + outputs['loss_consistency']
            loss_imputation += imp_loss.item()

            if task == 'C':
                BCE_f, _ = criterion(outputs['y_score_f'], outputs['y_out_f'], y.unsqueeze(1))
                BCE_b, _ = criterion(outputs['y_score_b'], outputs['y_out_b'], y.unsqueeze(1))
                cls_loss = BCE_f + BCE_b
                loss_classification += cls_loss.item()
                
                loss_total = (args.imputation_weight * imp_loss + 
                            args.classification_weight * cls_loss + 
                            args.consistency_weight * outputs['loss_consistency'])
            else:
                loss_total = (args.imputation_weight * imp_loss + 
                            args.consistency_weight * outputs['loss_consistency'])
            
            loss += loss_total.item()

            # Store data for metric calculation
            eval_m_all.append(eval_m.cpu().numpy())
            eval_x_all.append(eval_x.cpu().numpy())
            imp_all.append(outputs['imputation'].cpu().numpy())

            if task == 'C':
                y_true.extend(y.cpu().numpy())
                y_batch_score = (outputs['y_score_f'] + outputs['y_score_b']) / 2
                y_score.extend(y_batch_score.cpu().numpy().reshape(-1))
                y_pred.extend(np.round(y_batch_score.cpu().numpy()).reshape(-1))

        # Average losses
        num_batches = i + 1
        metrics = {
            'loss': loss / num_batches,
            'loss_imputation': loss_imputation / num_batches,
            'loss_classification': loss_classification / num_batches
        }

        # Calculate imputation metrics
        eval_x_all = np.concatenate(eval_x_all, axis=0)
        eval_m_all = np.concatenate(eval_m_all, axis=0)
        imp_all = np.concatenate(imp_all, axis=0)

        metrics['mae'] = np.sum(np.abs(eval_x_all - imp_all) * eval_m_all) / np.sum(eval_m_all)
        metrics['mre'] = (np.sum(np.abs(eval_x_all - imp_all) * eval_m_all) / 
                         np.sum(np.abs(eval_x_all) * eval_m_all))
        metrics['feature_mae'] = np.mean(np.abs(eval_x_all - imp_all) * eval_m_all, axis=(0,1))

        # Calculate classification metrics if applicable
        if task == 'C':
            cls_metrics = calculate_metrics(
                y_true=np.array(y_true),
                y_score=np.array(y_score),
                y_pred=np.array(y_pred)
            )
            metrics.update(cls_metrics)

        # Log metrics
        logger.log(f'------ {phase.capitalize()} Results ------')
        logger.log(f'Loss: {metrics["loss"]:.6f}')
        logger.log(f'Loss imputation: {metrics["loss_imputation"]:.6f}')
        if task == 'C':
            logger.log(f'Loss classification: {metrics["loss_classification"]:.6f}')
        logger.log(f'MAE: {metrics["mae"]:.6f}')
        logger.log(f'MRE: {metrics["mre"]:.6f}')

        if task == 'C':
            logger.log(f'Accuracy: {metrics["accuracy"]:.6f}')
            logger.log(f'AUC: {metrics["auc"]:.6f}')
            logger.log(f'Precision macro: {metrics["prec_macro"]:.6f}')
            logger.log(f'Recall macro: {metrics["recall_macro"]:.6f}')
            logger.log(f'F1 macro: {metrics["f1_macro"]:.6f}')
            logger.log(f'Balanced accuracy: {metrics["bal_acc"]:.6f}')

        # Log to tensorboard
        for tag, value in metrics.items():
            if isinstance(value, (int, float)):  # Skip non-scalar values
                tfw.add_scalar(tag=tag, scalar_value=value, global_step=epoch)

    return metrics