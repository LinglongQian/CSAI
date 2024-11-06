"""
Result processing script for CSAI experiments.

This script:
- Organizes experiment results 
- Processes datetime-based directories
- Aggregates results across experiments
- Creates summary statistics
"""

import os
import re
import pandas as pd
import pickle
import shutil
import argparse
from pathlib import Path
from typing import Optional, List, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process experimental results from CSAI experiments.'
    )
    
    # Directory configurations
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./log',
        help='Root directory containing experiment logs'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Directory to save processed results'
    )
    
    # Dataset configuration
    parser.add_argument(
        '--dataset',
        type=str,
        default='physionet',
        choices=['physionet', 'mimic_59f', 'eicu'],
        help='Dataset name'
    )
    
    # Processing configurations
    parser.add_argument(
        '--key_pattern',
        type=str,
        default='bets_valid',
        help='Pattern to match in result keys'
    )
    parser.add_argument(
        '--date_format',
        type=str,
        default=r'%Y%m%d\.%H\.%M\.%S',
        help='Format string for datetime directories'
    )
    
    return parser.parse_args()

class ResultProcessor:
    """Process and organize experimental results."""
    
    def __init__(self, args):
        """
        Initialize result processor.
        
        Args:
            args: Parsed command line arguments
        """
        self.root_path = Path(args.log_dir)
        self.dataset = args.dataset
        self.key_pattern = args.key_pattern
        self.results_dir = Path(args.results_dir) / args.dataset
        
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if not (self.results_dir / 'valid').exists():
            (self.results_dir / 'valid').mkdir(exist_ok=True)
        
        logger.info(f"Initialized ResultProcessor with:")
        logger.info(f"  Root path: {self.root_path}")
        logger.info(f"  Dataset: {self.dataset}")
        logger.info(f"  Key pattern: {self.key_pattern}")
        logger.info(f"  Results directory: {self.results_dir}")

    def find_and_process_directories(self, date_format: str):
        """
        Find and process datetime-based directories.
        
        Args:
            date_format: Format string for datetime directories
        """
        # Convert date format to regex pattern
        date_pattern = date_format.replace(
            '%Y', r'\d{4}'
        ).replace(
            '%m', r'\d{2}'
        ).replace(
            '%d', r'\d{2}'
        ).replace(
            '%H', r'\d{2}'
        ).replace(
            '%M', r'\d{2}'
        ).replace(
            '%S', r'\d{2}'
        )
        date_pattern = re.compile(date_pattern)
        
        # Walk through directories
        directory_path = self.root_path / self.dataset
        processed_count = 0
        
        for root, dirs, files in os.walk(directory_path):
            for dir_name in dirs:
                if date_pattern.match(dir_name):
                    full_dir_path = Path(root) / dir_name
                    
                    # Check for pickle files
                    if any(f.endswith('.pkl') for f in os.listdir(full_dir_path)):
                        logger.info(f"Processing directory: {full_dir_path}")
                        parent_dir = full_dir_path.parent
                        
                        # Record datetime in parent directory
                        with open(parent_dir / "exp_datetime.txt", "a") as log_file:
                            log_file.write(f"{dir_name}\n")
                        
                        try:
                            # Move contents to parent directory
                            for item in os.listdir(full_dir_path):
                                shutil.move(
                                    str(full_dir_path / item),
                                    str(parent_dir / item)
                                )
                            
                            # Remove empty directory
                            os.rmdir(full_dir_path)
                            processed_count += 1
                            
                        except Exception as e:
                            logger.error(f"Error processing directory {full_dir_path}: {e}")
        
        logger.info(f"Processed {processed_count} directories")

    def process_experiment_results(self):
        """Process results from all experiments."""
        exp_path = self.root_path / self.dataset
        if not exp_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {exp_path}")
        
        processed_models = 0
        total_models = len([d for d in os.listdir(exp_path) if (exp_path / d).is_dir()])
        
        # Process each experiment
        for model_name in os.listdir(exp_path):
            try:
                path = exp_path / model_name
                if not path.is_dir():
                    continue
                
                results = self._process_single_experiment(path)
                if results is not None and not results.empty:
                    # Sort and save results
                    results = results.sort_values(['model'], ascending=[False])
                    save_path = self.results_dir / 'valid' / f'{model_name}.csv'
                    results.to_csv(save_path, index=False)
                    logger.info(f"Saved results for {model_name} to {save_path}")
                    processed_models += 1
                
            except Exception as e:
                logger.error(f"Error processing {model_name}: {e}")
                continue
        
        logger.info(f"Successfully processed {processed_models}/{total_models} models")

    def _process_single_experiment(self, exp_path: Path) -> Optional[pd.DataFrame]:
        """
        Process results from a single experiment.
        
        Args:
            exp_path: Path to experiment directory
            
        Returns:
            DataFrame containing processed results, or None if processing fails
        """
        results = pd.DataFrame()
        processed_dirs = 0
        
        # Process each subdirectory in experiment
        for subdir in os.listdir(exp_path):
            try:
                result_path = exp_path / subdir / 'kfold_best.pkl'
                if not result_path.exists():
                    logger.warning(f"No results found in: {subdir}")
                    continue
                
                # Load and process results
                with open(result_path, 'rb') as f:
                    fold_results = pickle.load(f)
                
                subresults = pd.DataFrame()
                for key, value in fold_results.items():
                    if self.key_pattern in key:
                        value['model'] = subdir
                        value['fold'] = key
                        subresults = pd.concat([subresults, pd.DataFrame([value])])
                
                # Calculate overall metrics
                if not subresults.empty:
                    cols_to_use = subresults.columns.drop(['fold', 'model'])
                    overall = subresults[cols_to_use].mean().to_frame().T
                    overall['model'] = f"{subdir}_overall_valid"
                    subresults = pd.concat([subresults, overall])
                
                results = pd.concat([results, subresults])
                processed_dirs += 1
                
            except Exception as e:
                logger.warning(f"Error processing {subdir}: {e}")
                continue
        
        logger.info(f"Processed {processed_dirs} subdirectories for {exp_path.name}")
        return results

def main():
    """Main function for result processing."""
    # Parse arguments
    args = parse_args()
    
    # Initialize processor
    processor = ResultProcessor(args)
    
    # Process datetime directories
    logger.info("Processing datetime directories...")
    processor.find_and_process_directories(args.date_format)
    
    # Process experiment results
    logger.info("Processing experiment results...")
    processor.process_experiment_results()
    
    logger.info("Result processing completed successfully!")

if __name__ == '__main__':
    main()