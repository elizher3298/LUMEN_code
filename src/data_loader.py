"""
Data loading module for LUMEN model
Loads pre-processed CSV files and applies feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.feature import FeatureEngineer

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for LUMEN model"""
    
    def __init__(self, data_dir: str = str(Path(__file__).resolve().parent.parent / "data")):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing data CSV files
        """
        self.data_dir = Path(data_dir)
        self.feature_engineer = FeatureEngineer()
        
        # Data file paths
        self.train_file = self.data_dir / "training_korjap.csv"
        self.val_korjap_file = self.data_dir / "validation_korjap.csv"
        self.val_pangaea_file = self.data_dir / "validation_pangaea.csv"
        
        # Check if files exist
        self._check_files()
        
        # Cache for loaded data
        self._cache = {}
        
    def _check_files(self):
        """Check if required data files exist"""
        for file_path in [self.train_file, self.val_korjap_file, self.val_pangaea_file]:
            if not file_path.exists():
                logger.warning(f"Data file not found: {file_path}")
    
    def load_raw_data(self, dataset: str = 'train') -> pd.DataFrame:
        """
        Load raw data from CSV file
        
        Args:
            dataset: Which dataset to load ('train', 'val_korjap', 'val_pangaea')
            
        Returns:
            DataFrame with raw data
        """
        if dataset in self._cache:
            return self._cache[dataset].copy()
        
        file_map = {
            'train': self.train_file,
            'val_korjap': self.val_korjap_file,
            'val_pangaea': self.val_pangaea_file
        }
        
        if dataset not in file_map:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        file_path = file_map[dataset]
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        self._cache[dataset] = df
        logger.info(f"Loaded {len(df)} rows from {dataset}")
        
        return df.copy()
    
    def load_training_data(self, 
                          feature_set: str = 'basic',
                          target_name: str = 'srad',
                          validation_split: float = 0.2,
                          random_state: int = 42) -> Dict:
        """
        Load and prepare training data with train/val split
        
        Args:
            feature_set: Feature set to use
            target_name: Target variable name
            validation_split: Fraction of data to use for validation
            random_state: Random seed for splitting
            
        Returns:
            Dictionary with train/val data
        """
        # Load raw data
        df = self.load_raw_data('train')
        
        # Apply feature engineering
        X, y, metadata = self.feature_engineer.prepare_data(
            df, 
            feature_set=feature_set,
            target_name=target_name
        )
        
        # Split by stations to avoid data leakage
        unique_stations = metadata['station_id'].unique()
        train_stations, val_stations = train_test_split(
            unique_stations, 
            test_size=validation_split,
            random_state=random_state
        )
        
        # Create masks
        train_mask = metadata['station_id'].isin(train_stations)
        val_mask = metadata['station_id'].isin(val_stations)
        
        # Split data
        X_train = X[train_mask]
        X_val = X[val_mask]
        y_train = y[train_mask]
        y_val = y[val_mask]
        metadata_train = metadata[train_mask]
        metadata_val = metadata[val_mask]
        
        logger.info(f"Training set: {len(X_train)} samples from {len(train_stations)} stations")
        logger.info(f"Validation set: {len(X_val)} samples from {len(val_stations)} stations")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'metadata_train': metadata_train,
            'metadata_val': metadata_val,
            'feature_names': list(X.columns),
            'train_stations': train_stations,
            'val_stations': val_stations
        }
    
    def load_test_data(self, 
                      dataset: str = 'val_korjap',
                      feature_set: str = 'basic',
                      target_name: str = 'srad') -> Dict:
        """
        Load test/validation data
        
        Args:
            dataset: Which test dataset ('val_korjap' or 'val_pangaea')
            feature_set: Feature set to use
            target_name: Target variable name
            
        Returns:
            Dictionary with test data
        """
        # Load raw data
        df = self.load_raw_data(dataset)
        
        # Apply feature engineering
        X, y, metadata = self.feature_engineer.prepare_data(
            df,
            feature_set=feature_set,
            target_name=target_name
        )
        
        logger.info(f"Test set ({dataset}): {len(X)} samples from {metadata['station_id'].nunique()} stations")
        
        return {
            'X': X,
            'y': y,
            'metadata': metadata,
            'feature_names': list(X.columns)
        }
    
    def create_experiment_data(self,
                             feature_set: str,
                             target_name: str,
                             experiment_dir: str) -> Path:
        """
        Create and save experiment data files
        
        Args:
            feature_set: Feature set to use
            target_name: Target name
            experiment_dir: Directory to save experiment data
            
        Returns:
            Path to experiment directory
        """
        exp_path = Path(experiment_dir)
        exp_path.mkdir(parents=True, exist_ok=True)
        
        # Create experiment name
        exp_name = f"{feature_set}_{target_name}"
        exp_subdir = exp_path / exp_name
        exp_subdir.mkdir(exist_ok=True)
        
        logger.info(f"Creating experiment data in {exp_subdir}")
        
        # Process training data
        logger.info("Processing training data...")
        train_df = self.load_raw_data('train')
        X_train, y_train, metadata_train = self.feature_engineer.prepare_data(
            train_df,
            feature_set=feature_set,
            target_name=target_name
        )
        
        # Save training data
        train_input = pd.concat([metadata_train, X_train], axis=1)
        train_input['target'] = y_train
        train_input.to_csv(exp_subdir / 'train_data.csv', index=False)
        
        # Process Korea/Japan validation
        logger.info("Processing Korea/Japan validation data...")
        val_korjap_df = self.load_raw_data('val_korjap')
        X_val_kj, y_val_kj, metadata_val_kj = self.feature_engineer.prepare_data(
            val_korjap_df,
            feature_set=feature_set,
            target_name=target_name
        )
        
        # Save validation data
        val_kj_input = pd.concat([metadata_val_kj, X_val_kj], axis=1)
        val_kj_input['target'] = y_val_kj
        val_kj_input.to_csv(exp_subdir / 'val_korjap_data.csv', index=False)
        
        # Process PANGAEA validation
        logger.info("Processing PANGAEA validation data...")
        val_pangaea_df = self.load_raw_data('val_pangaea')
        X_val_pg, y_val_pg, metadata_val_pg = self.feature_engineer.prepare_data(
            val_pangaea_df,
            feature_set=feature_set,
            target_name=target_name
        )
        
        # Save PANGAEA validation data
        val_pg_input = pd.concat([metadata_val_pg, X_val_pg], axis=1)
        val_pg_input['target'] = y_val_pg
        val_pg_input.to_csv(exp_subdir / 'val_pangaea_data.csv', index=False)
        
        # Save configuration
        config = {
            'feature_set': feature_set,
            'target_name': target_name,
            'features': list(X_train.columns),
            'n_features': len(X_train.columns),
            'train_samples': len(X_train),
            'val_korjap_samples': len(X_val_kj),
            'val_pangaea_samples': len(X_val_pg),
            'train_stations': metadata_train['station_id'].nunique(),
            'val_korjap_stations': metadata_val_kj['station_id'].nunique(),
            'val_pangaea_stations': metadata_val_pg['station_id'].nunique()
        }
        
        import json
        with open(exp_subdir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Experiment data created successfully:")
        logger.info(f"  - Features: {config['features']}")
        logger.info(f"  - Training: {config['train_samples']} samples")
        logger.info(f"  - Val KorJap: {config['val_korjap_samples']} samples")
        logger.info(f"  - Val PANGAEA: {config['val_pangaea_samples']} samples")
        
        return exp_subdir
    
    def get_data_statistics(self, dataset: str = 'train') -> Dict:
        """
        Get statistics about a dataset
        
        Args:
            dataset: Dataset name
            
        Returns:
            Dictionary with statistics
        """
        df = self.load_raw_data(dataset)
        
        stats = {
            'n_samples': len(df),
            'n_stations': df['station_id'].nunique(),
            'date_range': {
                'start': str(df['date'].min()),
                'end': str(df['date'].max())
            },
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'srad_stats': {
                'mean': df['srad'].mean(),
                'std': df['srad'].std(),
                'min': df['srad'].min(),
                'max': df['srad'].max()
            }
        }
        
        # Station distribution
        station_counts = df['station_id'].value_counts()
        stats['station_distribution'] = {
            'mean_samples_per_station': station_counts.mean(),
            'min_samples': station_counts.min(),
            'max_samples': station_counts.max()
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize data loader
    loader = DataLoader()
    
    # Show data statistics
    print("\nData Statistics:")
    for dataset in ['train', 'val_korjap', 'val_pangaea']:
        try:
            stats = loader.get_data_statistics(dataset)
            print(f"\n{dataset}:")
            print(f"  Samples: {stats['n_samples']}")
            print(f"  Stations: {stats['n_stations']}")
            print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        except Exception as e:
            print(f"  Error loading {dataset}: {e}")
    
    # Create experiment data for different feature sets
    print("\nCreating experiment data...")
    experiment_dir = str(Path(__file__).resolve().parent.parent / "experiments")
    
    for feature_set in ['basic', 'normalized']:
        for target in ['srad', 'srad_norm']:
            try:
                exp_path = loader.create_experiment_data(
                    feature_set=feature_set,
                    target_name=target,
                    experiment_dir=experiment_dir
                )
                print(f"Created: {exp_path}")
            except Exception as e:
                print(f"Error creating {feature_set}_{target}: {e}")
