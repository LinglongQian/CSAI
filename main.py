"""
Main training script for CSAI model.

This script handles:
- Dataset loading and preprocessing
- Model training and evaluation
- Cross-validation experiments
- Result saving and visualization
"""

import os
import pickle
import datetime
import argparse
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

from utils import (
    setup_seed, normalize, calculate_metrics,
    get_polarfig, ExperimentLogger, save_training_info,
    non_uniform_sample_loader_bidirectional, evaluate, config
)
from losses import DiceBCELoss

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CSAI Training Script')
    
    # Hardware configs
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--seed", type=int, default=1)
    
    # Model configs
    parser.add_argument("--model_name", type=str, default='CSAI')
    parser.add_argument("--hiddens", type=int, default=108)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--step_channels", type=int, default=512)
    parser.add_argument("--pre_model", type=str, default='.')
    
    # Dataset configs
    parser.add_argument("--dataset", type=str, default='physionet')
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--removal_percent", type=int, default=10)
    
    # Training configs
    parser.add_argument("--task", type=str, default='I')
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    
    # Loss weights
    parser.add_argument("--imputation_weight", type=float, default=0.3)
    parser.add_argument("--classification_weight", type=float, default=1)
    parser.add_argument("--consistency_weight", type=float, default=0.1)
    parser.add_argument("--increase_factor", type=float, default=0.5)
    
    # Output configs
    parser.add_argument("--model_path", type=str, default='./log')
    parser.add_argument("--out_size", type=int, default=1)
    
    args = parser.parse_args()
    return args

def setup_environment(args):
    """Setup GPU and random seeds."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(args.seed)
    return args

def load_data(args) -> Tuple[List, List]:
    """Load and verify dataset."""
    try:
        kfold_data = pickle.load(open(args.data_path, 'rb'))
        kfold_label = pickle.load(open(args.label_path, 'rb'))
        return kfold_data, kfold_label
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")
    
def setup_fold_tracking(args, fold: int) -> Tuple[ExperimentLogger, Dict[str, SummaryWriter], Path]:
    """Setup logging and tracking for a fold."""
    date_str = datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S')
    fold_dir = Path(args.model_path) / args.dataset / \
               f'{args.model_name}_remove_{args.removal_percent}' / \
               f'task_{args.task}' / date_str / f'fold_{fold}'
    
    # Create directories
    for subdir in ['tflog', 'model_state']:
        (fold_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = ExperimentLogger(fold_dir / 'log.txt')
    logger.log_config(args)
    
    # Setup tensorboard writers
    writers = {
        'train': SummaryWriter(fold_dir / 'tflog/train'),
        'valid': SummaryWriter(fold_dir / 'tflog/valid'),
        'test': SummaryWriter(fold_dir / 'tflog/test')
    }
    
    return logger, writers, fold_dir

def prepare_fold_data(
    fold: int,
    kfold_data: List,
    kfold_label: List,
    args
) -> Tuple[Dict[str, torch.utils.data.DataLoader], Dict[str, float]]:
    """Prepare data loaders for a fold."""
    # Get fold data
    train_data, valid_data, test_data = kfold_data[fold]
    train_label, valid_label, test_label = kfold_label[fold]
    
    # Log data statistics
    print('Unbalanced ratios:')
    print(f'Train: {sum(train_label)/len(train_label):.3f}')
    print(f'Valid: {sum(valid_label)/len(valid_label):.3f}')
    print(f'Test: {sum(test_label)/len(test_label):.3f}')
    
    # Normalize data
    train_data, mean_set, std_set, intervals = normalize(
        data=train_data,
        mean=[],
        std=[],
        compute_intervals=True
    )
    valid_data, _, _ = normalize(valid_data, mean_set, std_set)
    test_data, _, _ = normalize(test_data, mean_set, std_set)
    
    # Calculate missing rates
    missing_rates = {
        'train': np.isnan(train_data).sum(axis=(0,1)) / (train_data.shape[0] * train_data.shape[1]),
        'valid': np.isnan(valid_data).sum(axis=(0,1)) / (valid_data.shape[0] * valid_data.shape[1]),
        'test': np.isnan(test_data).sum(axis=(0,1)) / (test_data.shape[0] * test_data.shape[1])
    }
    
    # Create data loaders
    train_loader, replacement_probs = non_uniform_sample_loader_bidirectional(
        train_data, train_label, args.batchsize, args.removal_percent, 
        increase_factor=args.increase_factor
    )
    
    valid_loader, _ = non_uniform_sample_loader_bidirectional(
        valid_data, valid_label, args.batchsize, args.removal_percent, 
        pre_replacement_probabilities=replacement_probs
    )
    
    test_loader, _ = non_uniform_sample_loader_bidirectional(
        test_data, test_label, args.batchsize, args.removal_percent,
        pre_replacement_probabilities=replacement_probs
    )
    
    loaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }
    
    return loaders, missing_rates, intervals, replacement_probs

def initialize_model(args, intervals):
    """Initialize model, criterion, and optimizer."""
    # Import appropriate model
    if args.model_name == 'CSAI':
        from models import bcsai as net
    elif args.model_name == 'Brits':
        from models import brits as net
    elif args.model_name == 'Brits_gru':
        from models import brits_gru as net  
    elif args.model_name == 'GRUD':
        from models import gru_d as net
    elif args.model_name == 'BVRIN':
        from models import bvrin as net
    elif args.model_name == 'MRNN':
        from models import m_rnn as net
    else:
        raise ValueError(f"Unknown model: {args.model_name}")

    # Initialize model
    model = net(args=args, medians_df=intervals, get_y=(args.task == 'C')).to(args.device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params:,}')
    
    # Initialize criterion and optimizer
    criterion = DiceBCELoss().to(args.device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    return model, criterion, optimizer

def train_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    args,
    epoch: int,
    phase: str,
    logger: ExperimentLogger,
    writer: SummaryWriter
) -> Dict:
    """Train for one epoch."""
    model.train()
    
    metrics = {
        'loss': 0,
        'loss_imputation': 0,
        'loss_classification': 0,
    }
    
    all_y = []
    all_y_pred = []
    all_y_score = []
    eval_x_all = []
    eval_m_all = []
    imp_all = []
    
    for i, batch in enumerate(loader):
        # Move data to device
        y = batch['labels'].to(args.device)
        eval_x = batch['evals'].to(args.device)
        eval_m = batch['eval_masks'].to(args.device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch)
        
        # Calculate losses
        imp_loss = outputs['loss_regression'] + outputs['loss_consistency']
        metrics['loss_imputation'] += imp_loss.item()
        
        if args.task == 'C':
            BCE_f, _ = criterion(outputs['y_score_f'], outputs['y_out_f'], y.unsqueeze(1))
            BCE_b, _ = criterion(outputs['y_score_b'], outputs['y_out_b'], y.unsqueeze(1))
            cls_loss = BCE_f + BCE_b
            metrics['loss_classification'] += cls_loss.item()
            
            loss = (args.imputation_weight * imp_loss + 
                   args.classification_weight * cls_loss + 
                   args.consistency_weight * outputs['loss_consistency'])
        else:
            loss = (args.imputation_weight * imp_loss + 
                   args.consistency_weight * outputs['loss_consistency'])
        
        metrics['loss'] += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Collect data for metrics
        eval_x_all.append(eval_x.cpu().numpy())
        eval_m_all.append(eval_m.cpu().numpy())
        imp_all.append(outputs['imputation'].detach().cpu().numpy())
        
        if args.task == 'C':
            all_y.append(y.cpu().numpy())
            y_score = (outputs['y_score_f'] + outputs['y_score_b']) / 2
            all_y_score.append(y_score.detach().cpu().numpy())
            all_y_pred.append(np.round(y_score.detach().cpu().numpy()))
    
    # Calculate final metrics
    metrics = {k: v / (i + 1) for k, v in metrics.items()}
    
    # Calculate imputation metrics
    eval_x_all = np.concatenate(eval_x_all)
    eval_m_all = np.concatenate(eval_m_all)
    imp_all = np.concatenate(imp_all)
    
    metrics['mae'] = np.sum(np.abs(eval_x_all - imp_all) * eval_m_all) / np.sum(eval_m_all)
    metrics['mre'] = (np.sum(np.abs(eval_x_all - imp_all) * eval_m_all) / 
                     np.sum(np.abs(eval_x_all) * eval_m_all))
    metrics['feature_mae'] = np.mean(np.abs(eval_x_all - imp_all) * eval_m_all, axis=(0,1))
    
    # Calculate classification metrics
    if args.task == 'C':
        all_y = np.concatenate(all_y)
        all_y_score = np.concatenate(all_y_score)
        all_y_pred = np.concatenate(all_y_pred)
        
        cls_metrics = calculate_metrics(all_y, all_y_score, all_y_pred)
        metrics.update(cls_metrics)
    
    # Log metrics
    logger.log(f'Loss: {metrics["loss"]:.6f}')
    logger.log(f'MAE: {metrics["mae"]:.6f}')
    if args.task == 'C':
        logger.log(f'AUC: {metrics["auc"]:.6f}')
    
    # Log to tensorboard
    if writer is not None:
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f'{phase}/{name}', value, epoch)
    
    return metrics

def train_fold(
    fold: int,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loaders: Dict[str, torch.utils.data.DataLoader],
    args,
    logger: ExperimentLogger,
    writers: Dict[str, SummaryWriter],
    save_dir: Path,
    missing_rates: Dict[str, np.ndarray],
    replacement_probs: np.ndarray
) -> Dict:
    """Train and evaluate a single fold."""
    best_metrics = {
        'train': {'epoch': 0, 'mae': float('inf'), 'auc': 0},
        'valid': {'epoch': 0, 'mae': float('inf'), 'auc': 0},
        'test': {'epoch': 0, 'mae': float('inf'), 'auc': 0},
    }

    train_info = {
        'Loss': [], 'Loss_imputation': [], 'Loss_classification': [],
        'Mae': [], 'Mre': [], 'Auc': [], 'prec_macro': [],
        'recall_macro': [], 'f1_macro': [], 'bal_acc': []
    }
    valid_info = copy.deepcopy(train_info)
    test_info = copy.deepcopy(train_info)

    # Training loop
    for epoch in range(args.epoch):
        logger.log(f'\n------ Epoch {epoch + 1}/{args.epoch}')
        
        # Training phase
        logger.log('-- Training')
        train_metrics = train_epoch(
            model=model,
            criterion=criterion,
            loader=loaders['train'],
            optimizer=optimizer,
            args=args,
            epoch=epoch,
            phase='train',
            logger=logger,
            writer=writers['train']
        )

        # Store training info
        for key in train_info.keys():
            if key in train_metrics:
                train_info[key].append(train_metrics[key])

        # Validation phase
        logger.log('-- Validation')
        valid_metrics = evaluate(
            phase='valid',
            model=model,
            criterion=criterion,
            data=loaders['valid'],
            args=args,
            task=args.task,
            logger=logger,
            tfw=writers['valid'],
            epoch=epoch
        )

        # Store validation info
        for key in valid_info.keys():
            if key in valid_metrics:
                valid_info[key].append(valid_metrics[key])

        # Testing phase
        logger.log('-- Testing')
        test_metrics = evaluate(
            phase='test',
            model=model,
            criterion=criterion,
            data=loaders['test'],
            args=args,
            task=args.task,
            logger=logger,
            tfw=writers['test'],
            epoch=epoch
        )
        
        # Store test info
        for key in test_info.keys():
            if key in test_metrics:
                test_info[key].append(test_metrics[key])

        # Save best models and update metrics
        for phase, metrics in zip(['train', 'valid', 'test'], 
                                [train_metrics, valid_metrics, test_metrics]):
            # Check if current model is best
            is_best = False
            if args.task == 'I' and metrics['mae'] < best_metrics[phase]['mae']:
                is_best = True
                best_metrics[phase].update({
                    'epoch': epoch,
                    'mae': metrics['mae'],
                    'mre': metrics['mre']
                })
                
                # Save visualization
                get_polarfig(
                    args, replacement_probs,
                    missing_rates[phase],
                    metrics['feature_mae'],
                    save_dir, fold, phase,
                    args.attributes
                )
                
                # Log best metrics for imputation task
                logger.log(f'Best {phase} metrics found!')
                logger.log(f'MAE: {metrics["mae"]:.6f}')
                logger.log(f'MRE: {metrics["mre"]:.6f}')
                
            elif args.task == 'C' and metrics['auc'] > best_metrics[phase]['auc']:
                is_best = True
                best_metrics[phase].update({
                    'epoch': epoch,
                    'mae': metrics['mae'],
                    'mre': metrics['mre'],
                    'accuracy': metrics['accuracy'],
                    'auc': metrics['auc'],
                    'prec_macro': metrics['prec_macro'],
                    'recall_macro': metrics['recall_macro'],
                    'f1_macro': metrics['f1_macro'],
                    'bal_acc': metrics['bal_acc']
                })
                
                # Save visualization
                get_polarfig(
                    args, replacement_probs,
                    missing_rates[phase],
                    metrics['feature_mae'],
                    save_dir, fold, phase,
                    args.attributes
                )
                
                # Log best metrics for classification task
                logger.log(f'Best {phase} metrics found!')
                logger.log(f'AUC: {metrics["auc"]:.6f}')
                logger.log(f'Accuracy: {metrics["accuracy"]:.6f}')
                logger.log(f'MAE: {metrics["mae"]:.6f}')
            
            if is_best:
                # Save model
                save_path = save_dir / 'model_state' / f'model_{fold}_best_{phase}_{args.task}_state_dict.pth'
                torch.save(model.state_dict(), save_path)
                logger.log(f'Saved best {phase} model to {save_path}')
    
    # Save complete training info
    training_record = {
        'train': train_info,
        'valid': valid_info,
        'test': test_info
    }
    
    save_training_info(
        info=training_record,
        save_dir=save_dir,
        fold=fold,
        args=args
    )
    
    return best_metrics

def main():
    """Main training function."""
    # Parse arguments and setup
    args = parse_args()
    args = setup_environment(args)

    # Configure dataset-specific parameters
    args = config(args)  # From utils.py

    # Set paths
    args.data_path = f'./data/{args.dataset}/data_nan.pkl'
    args.label_path = f'./data/{args.dataset}/label.pkl'
    
    # Load data
    kfold_data, kfold_label = load_data(args)
    
    # Store results for all folds
    results = {}
    
    # Training loop for each fold
    for fold in range(5):
        print(f'\nProcessing Fold {fold+1}/5')
        
        # Setup tracking for this fold
        logger, writers, save_dir = setup_fold_tracking(args, fold)
        
        # Prepare data
        loaders, missing_rates, intervals, replacement_probs = prepare_fold_data(
            fold, kfold_data, kfold_label, args
        )
        
        # Initialize model
        model, criterion, optimizer = initialize_model(args, intervals)
        
        # Train fold
        fold_metrics = train_fold(
            fold=fold,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            args=args,
            logger=logger,
            writers=writers,
            save_dir=save_dir,
            missing_rates=missing_rates,
            replacement_probs=replacement_probs
        )
        
        # Store results
        results[f'fold_{fold}'] = fold_metrics
        
        # Close writers
        for writer in writers.values():
            writer.close()
        
        # Log final results for fold
        logger.log('\nFinal Best Results:')
        for phase in ['train', 'valid', 'test']:
            logger.log(f'\n{phase.upper()} METRICS:')
            for metric, value in fold_metrics[phase].items():
                if isinstance(value, (int, float)):
                    logger.log(f'{metric}: {value:.6f}')
    
    # Save overall results
    result_path = Path(args.model_path) / args.dataset / \
                 f'{args.model_name}_remove_{args.removal_percent}' / \
                 f'task_{args.task}' / 'kfold_best.pkl'
    
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'wb') as f:
        pickle.dump(results, f, protocol=-1)
    
    print('\nTraining completed successfully!')

if __name__ == '__main__':
    main()









