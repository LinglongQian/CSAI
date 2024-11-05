#!/usr/bin/env python3
"""
PhysioNet Data Processor

This script processes raw PhysioNet Challenge 2012 data into a format suitable for time series analysis.
It handles data loading, preprocessing, and k-fold splitting of the dataset.

"""
import os
import re
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

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
    return parser.parse_args()

def find_patient_ids(data_dir: Path) -> List[str]:
    """Extract valid patient IDs from filenames."""
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
    """Load patient outcome data."""
    outcomes_path = data_dir / 'Outcomes-a.txt'
    if not outcomes_path.exists():
        raise FileNotFoundError(f"Outcomes file not found: {outcomes_path}")
    
    try:
        return pd.read_csv(outcomes_path).set_index('RecordID')['In-hospital_death']
    except Exception as e:
        raise ValueError(f"Failed to load outcomes data: {e}")

def parse_time(time_str: str) -> int:
    """Convert time string to hour bin."""
    try:
        hour, _ = map(int, time_str.split(':'))
        return hour
    except ValueError as e:
        raise ValueError(f"Invalid time format: {time_str}") from e
    
def parse_record(record: pd.DataFrame) -> List[float]:
    """Extract values for selected attributes from a patient record."""
    record = record.set_index('Parameter')['Value']
    values = []
    
    for attr in SELECTED_ATTRIBUTES:
        if attr in record.index:
            values.append(record[attr].mean())
        else:
            values.append(np.nan)
            
    return values

def generate_datasets(data_dir: Path, patient_ids: List[str], id_list: List[int],
                     outcomes: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Generate datasets for given patient IDs."""
    if not id_list:
        raise ValueError("Empty ID list provided")

    dataset = []
    labels = []
    
    for id_idx in tqdm(id_list, desc="Processing patients"):
        try:
            patient_id = patient_ids[id_idx]
            data = pd.read_csv(data_dir / f"{patient_id}.txt")
            
            # Group by hour and parse values
            data['Time'] = data['Time'].apply(parse_time)
            values = []
            
            for hour in range(48):
                hour_data = data[data['Time'] == hour]
                values.append(parse_record(hour_data))
            
            # Get outcome label
            labels.append(outcomes.loc[int(patient_id)])
            dataset.append(np.array(values))
            
        except Exception as e:
            logger.warning(f"Error processing patient {patient_id}: {e}")
            continue

    if not dataset:
        raise ValueError("No valid data generated")
        
    return np.array(dataset), np.array(labels)

def process_kfold(data_dir: Path, output_dir: Path, patient_ids: List[str],
                 outcomes: pd.Series, n_splits: int, seed: int):
    """Process data using k-fold cross validation."""
    kfold_data = []
    kfold_label = []
    
    # Initialize K-fold splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Process each fold
    for fold, (train_eval_idx, test_idx) in enumerate(kf.split(patient_ids)):
        logger.info(f"Processing fold {fold + 1}/{n_splits}")
        
        # Split train_eval into train and validation
        np.random.shuffle(train_eval_idx)
        val_size = len(test_idx)
        val_idx = train_eval_idx[:val_size]
        train_idx = train_eval_idx[val_size:]
        
        # Generate datasets
        train_data, train_label = generate_datasets(data_dir, patient_ids, train_idx, outcomes)
        val_data, val_label = generate_datasets(data_dir, patient_ids, val_idx, outcomes)
        test_data, test_label = generate_datasets(data_dir, patient_ids, test_idx, outcomes)
        
        kfold_data.append([train_data, val_data, test_data])
        kfold_label.append([train_label, val_label, test_label])
        
        # Log fold statistics
        logger.info(f"Fold {fold + 1} stats:")
        logger.info(f"Train samples: {len(train_data)}")
        logger.info(f"Val samples: {len(val_data)}")
        logger.info(f"Test samples: {len(test_data)}")
        logger.info(f"Positive ratio - Train: {train_label.mean():.3f}, "
                  f"Val: {val_label.mean():.3f}, Test: {test_label.mean():.3f}")

    # Save processed data
    try:
        pickle.dump(kfold_data, open(output_dir / 'data_nan.pkl', 'wb'), -1)
        pickle.dump(kfold_label, open(output_dir / 'label.pkl', 'wb'), -1)
        logger.info(f"Saved processed data to {output_dir}")
    except Exception as e:
        raise IOError(f"Failed to save processed data: {e}")

def main():
    """Main entry point for data processing."""
    args = parse_args()
    
    # Convert paths to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Validate directories
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
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
            seed=args.seed
        )
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise

if __name__ == '__main__':
    main()

