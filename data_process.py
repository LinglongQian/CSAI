#!/usr/bin/env python3
"""
PhysioNet Data Processor

This script processes raw PhysioNet Challenge 2012 data into a format suitable for time series analysis.
Features:
- Parallel processing support for large datasets
- Progress tracking and logging
- Robust error handling and validation
- Memory-efficient data handling
"""
import os
import re
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from functools import partial
import multiprocessing as mp
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# we select 35 attributes which contains enough non-values
# Selected attributes based on sufficient non-missing values
SELECTED_ATTRIBUTES = [
    'DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 
    'Albumin', 'ALT', 'Glucose', 'SaO2', 'Temp', 'AST', 'Bilirubin', 'HCO3', 
    'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS', 'Cholesterol',
    'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine',
    'NIMAP', 'Creatinine', 'ALP'
]

# List of patient IDs known to have no data
EXCLUDED_PATIENTS = ['141264', '140936', '140501']

class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass

@contextmanager
def get_process_pool(n_workers: Optional[int] = None) -> mp.Pool:
    """
    Context manager for process pool.
    
    Args:
        n_workers: Number of worker processes. If None, uses CPU count - 1
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    pool = mp.Pool(processes=n_workers)
    try:
        yield pool
    finally:
        pool.close()
        pool.join()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process PhysioNet Challenge 2012 data for time series analysis.'
    )
    parser.add_argument(
        '--data_dir', 
        type=str,
        default='./data/physionet',
        help='Directory containing raw PhysioNet data files'
    )
    parser.add_argument(
        '--output_dir', 
        type=str,
        default='./data/physionet',
        help='Directory to save processed data files'
    )
    parser.add_argument(
        '--n_splits',
        type=int,
        default=5,
        help='Number of folds for cross-validation'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=3407,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        default=None,
        help='Number of worker processes. Defaults to CPU count - 1'
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=10,
        help='Number of patients to process in each parallel chunk'
    )
    return parser.parse_args()

def validate_directories(data_dir: Path, output_dir: Path):
    """Validate input and output directories."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Check for required input files
    if not (data_dir / 'Outcomes-a.txt').exists():
        raise FileNotFoundError("Outcomes file not found")
        
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

def find_patient_ids(data_dir: Path) -> List[str]:
    """
    Extract valid patient IDs from filenames.
    
    Args:
        data_dir: Directory containing patient data files
        
    Returns:
        List of valid patient IDs
    """
    pattern = re.compile(r'\d{6}')
    patient_ids = []
    
    for filename in os.listdir(data_dir):
        match = pattern.search(filename)
        if match:
            patient_id = match.group()
            if patient_id not in EXCLUDED_PATIENTS:
                patient_ids.append(patient_id)
    
    if not patient_ids:
        raise ValueError("No valid patient IDs found in data directory")
        
    logger.info(f"Found {len(patient_ids)} valid patient records")
    return patient_ids

def load_outcomes(data_dir: Path) -> pd.Series:
    """
    Load patient outcome data.
    
    Args:
        data_dir: Directory containing outcomes file
        
    Returns:
        Series mapping patient IDs to outcomes
    """
    outcomes_path = data_dir / 'Outcomes-a.txt'
    try:
        outcomes = pd.read_csv(outcomes_path).set_index('RecordID')['In-hospital_death']
        logger.info(f"Loaded outcomes for {len(outcomes)} patients")
        return outcomes
    except Exception as e:
        raise DataProcessingError(f"Failed to load outcomes data: {e}")

def process_patient_record(args: Tuple[Path, str, pd.Series]) -> Optional[Tuple[np.ndarray, float]]:
    """
    Process a single patient record.
    
    Args:
        args: Tuple of (data_dir, patient_id, outcomes)
        
    Returns:
        Tuple of (processed_data, outcome_label) or None if processing fails
    """
    data_dir, patient_id, outcomes = args
    
    try:
        # Load patient data
        data = pd.read_csv(data_dir / f"{patient_id}.txt")
        
        # Convert time to hour bins
        data['Time'] = data['Time'].apply(lambda x: int(x.split(':')[0]))
        
        # Process each hour
        values = []
        for hour in range(48):
            hour_data = data[data['Time'] == hour]
            
            if hour_data.empty:
                values.append([np.nan] * len(SELECTED_ATTRIBUTES))
            else:
                # Extract values for selected attributes
                hour_values = []
                hour_data_indexed = hour_data.set_index('Parameter')['Value']
                
                for attr in SELECTED_ATTRIBUTES:
                    if attr in hour_data_indexed.index:
                        hour_values.append(hour_data_indexed[attr].mean())
                    else:
                        hour_values.append(np.nan)
                        
                values.append(hour_values)
        
        processed_data = np.array(values)
        outcome_label = outcomes.loc[int(patient_id)]
        
        return processed_data, outcome_label
        
    except Exception as e:
        logger.warning(f"Error processing patient {patient_id}: {e}")
        return None

def process_patient_chunk(patient_chunk: List[str], data_dir: Path, 
                         outcomes: pd.Series) -> Tuple[List[np.ndarray], List[float]]:
    """
    Process a chunk of patient records.
    
    Args:
        patient_chunk: List of patient IDs to process
        data_dir: Directory containing patient data
        outcomes: Series of patient outcomes
        
    Returns:
        Tuple of (processed_data_list, outcome_labels)
    """
    processed_data = []
    outcome_labels = []
    
    for patient_id in patient_chunk:
        result = process_patient_record((data_dir, patient_id, outcomes))
        if result is not None:
            data, label = result
            processed_data.append(data)
            outcome_labels.append(label)
            
    return processed_data, outcome_labels

def generate_datasets(data_dir: Path, patient_ids: List[str], id_list: List[int],
                     outcomes: pd.Series, n_workers: Optional[int] = None,
                     chunk_size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate datasets for given patient IDs using parallel processing.
    
    Args:
        data_dir: Directory containing patient data
        patient_ids: List of all patient IDs
        id_list: List of indices into patient_ids to process
        outcomes: Series of patient outcomes
        n_workers: Number of worker processes
        chunk_size: Number of patients to process in each chunk
        
    Returns:
        Tuple of (processed_data, outcome_labels)
    """
    if not id_list:
        raise DataProcessingError("Empty ID list provided")

    # Get patient IDs for this split
    split_patient_ids = [patient_ids[idx] for idx in id_list]
    
    # Create chunks of patient IDs
    chunks = [split_patient_ids[i:i + chunk_size] 
             for i in range(0, len(split_patient_ids), chunk_size)]
    
    all_data = []
    all_labels = []
    
    with get_process_pool(n_workers) as pool:
        # Process chunks in parallel
        process_func = partial(process_patient_chunk, data_dir=data_dir, outcomes=outcomes)
        
        for chunk_data, chunk_labels in tqdm(
            pool.imap(process_func, chunks),
            total=len(chunks),
            desc="Processing patient chunks"
        ):
            all_data.extend(chunk_data)
            all_labels.extend(chunk_labels)

    if not all_data:
        raise DataProcessingError("No valid data generated")
        
    return np.array(all_data), np.array(all_labels)

def process_kfold(data_dir: Path, output_dir: Path, patient_ids: List[str],
                 outcomes: pd.Series, n_splits: int, seed: int, 
                 n_workers: Optional[int] = None, chunk_size: int = 10):
    """
    Process data using k-fold cross validation with parallel processing.
    
    Args:
        data_dir: Directory containing raw data
        output_dir: Directory to save processed data
        patient_ids: List of patient IDs
        outcomes: Series of patient outcomes
        n_splits: Number of cross-validation folds
        seed: Random seed
        n_workers: Number of worker processes
        chunk_size: Number of patients to process in each chunk
    """
    kfold_data = []
    kfold_label = []
    
    # Initialize K-fold splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Process each fold
    for fold, (train_eval_idx, test_idx) in enumerate(kf.split(patient_ids), 1):
        logger.info(f"Processing fold {fold}/{n_splits}")
        
        # Split train_eval into train and validation
        np.random.shuffle(train_eval_idx)
        val_size = len(test_idx)
        val_idx = train_eval_idx[:val_size]
        train_idx = train_eval_idx[val_size:]
        
        # Generate datasets
        logger.info("Processing training data...")
        train_data, train_label = generate_datasets(
            data_dir, patient_ids, train_idx, outcomes, 
            n_workers, chunk_size
        )
        
        logger.info("Processing validation data...")
        val_data, val_label = generate_datasets(
            data_dir, patient_ids, val_idx, outcomes,
            n_workers, chunk_size
        )
        
        logger.info("Processing test data...")
        test_data, test_label = generate_datasets(
            data_dir, patient_ids, test_idx, outcomes,
            n_workers, chunk_size
        )
        
        kfold_data.append([train_data, val_data, test_data])
        kfold_label.append([train_label, val_label, test_label])
        
        # Log fold statistics
        logger.info(f"Fold {fold} statistics:")
        logger.info(f"Train samples: {len(train_data)}")
        logger.info(f"Val samples: {len(val_data)}")
        logger.info(f"Test samples: {len(test_data)}")
        logger.info(f"Positive ratio - Train: {train_label.mean():.3f}, "
                   f"Val: {val_label.mean():.3f}, Test: {test_label.mean():.3f}")
        
        # Calculate and log missingness statistics
        for name, data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
            missing_rates = np.isnan(data).mean(axis=(0, 1))
            logger.info(f"\n{name} set missingness rates:")
            for attr, rate in zip(SELECTED_ATTRIBUTES, missing_rates):
                logger.info(f"{attr}: {rate:.3f}")

    # Save processed data
    try:
        with open(output_dir / 'data_nan.pkl', 'wb') as f:
            pickle.dump(kfold_data, f, protocol=-1)
        with open(output_dir / 'label.pkl', 'wb') as f:
            pickle.dump(kfold_label, f, protocol=-1)
        logger.info(f"Saved processed data to {output_dir}")
    except Exception as e:
        raise DataProcessingError(f"Failed to save processed data: {e}")

def main():
    """Main entry point for data processing."""
    args = parse_args()
    
    # Convert paths to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    try:
        # Validate directories
        validate_directories(data_dir, output_dir)
        
        # Load data
        patient_ids = find_patient_ids(data_dir)
        outcomes = load_outcomes(data_dir)
        
        # Process data
        process_kfold(
            data_dir=data_dir,
            output_dir=output_dir,
            patient_ids=patient_ids,
            outcomes=outcomes,
            n_splits=args.n_splits,
            seed=args.seed,
            n_workers=args.n_workers,
            chunk_size=args.chunk_size
        )
        
        logger.info("Data processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise

if __name__ == '__main__':
    main()