"""
Experiment manager for LUMEN model
Manages experiments based on gpu_experiment_plan.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
from datetime import datetime

from src.feature import FeatureEngineer
from src.data_loader import DataLoader

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Manages experiments based on GPU experiment plan"""
    
    def __init__(self, 
                 experiment_plan_path: str = "configs/gpu_experiment_plan.csv",
                 data_dir: str = str(Path(__file__).resolve().parent.parent / "data"),
                 experiment_dir: str = str(Path(__file__).resolve().parent.parent / "experiments")):
        """
        Initialize experiment manager
        
        Args:
            experiment_plan_path: Path to GPU experiment plan CSV
            data_dir: Directory with data files
            experiment_dir: Directory to save experiments
        """
        self.experiment_plan_path = Path(experiment_plan_path)
        self.experiment_dir = Path(experiment_dir)
        self.data_loader = DataLoader(data_dir)
        self.feature_engineer = FeatureEngineer()
        
        # Load experiment plan
        self.experiment_plan = self._load_experiment_plan()
        
    def _load_experiment_plan(self) -> pd.DataFrame:
        """Load experiment plan from CSV"""
        if not self.experiment_plan_path.exists():
            raise FileNotFoundError(f"Experiment plan not found: {self.experiment_plan_path}")
        
        df = pd.read_csv(self.experiment_plan_path)
        logger.info(f"Loaded {len(df)} experiments from {self.experiment_plan_path}")
        
        # Ensure required columns exist
        required_cols = ['id', 'features']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column in experiment plan: {col}")
        
        # Add default values if columns missing
        if 'target' not in df.columns:
            df['target'] = 'srad'  # Default target
        if 'normalization' not in df.columns:
            df['normalization'] = 'standard'
        
        return df
    
    def get_experiment(self, experiment_id: int) -> Dict:
        """
        Get experiment configuration by ID
        
        Args:
            experiment_id: Experiment ID from the plan
            
        Returns:
            Dictionary with experiment configuration
        """
        exp_row = self.experiment_plan[self.experiment_plan['id'] == experiment_id]
        
        if exp_row.empty:
            raise ValueError(f"Experiment ID {experiment_id} not found")
        
        exp = exp_row.iloc[0]
        
        # Parse features (space-separated string to list)
        feature_list = exp['features'].split()
        
        config = {
            'id': int(exp['id']),
            'features': feature_list,
            'target': exp.get('target', 'srad'),
            'normalization': exp.get('normalization', 'standard')
        }
        
        return config
    
    def prepare_experiment_data(self, experiment_id: int) -> Path:
        """
        Prepare data for a specific experiment
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Path to experiment directory with prepared data
        """
        # Get experiment configuration
        exp_config = self.get_experiment(experiment_id)
        
        # Create experiment directory
        exp_name = f"exp_{experiment_id:03d}_{exp_config['target']}"
        exp_path = self.experiment_dir / exp_name
        exp_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Preparing experiment {experiment_id}")
        logger.info(f"Features: {exp_config['features']}")
        logger.info(f"Target: {exp_config['target']}")
        
        # Process training data
        logger.info("Processing training data...")
        train_df = self.data_loader.load_raw_data('train')
        X_train, y_train, metadata_train = self.feature_engineer.prepare_data(
            train_df,
            feature_list=exp_config['features'],
            target_name=exp_config['target']
        )
        
        # Save training data
        self._save_dataset(X_train, y_train, metadata_train, 
                          exp_path / 'train_data.csv')
        
        # Process Korea/Japan validation
        logger.info("Processing Korea/Japan validation data...")
        val_korjap_df = self.data_loader.load_raw_data('val_korjap')
        X_val_kj, y_val_kj, metadata_val_kj = self.feature_engineer.prepare_data(
            val_korjap_df,
            feature_list=exp_config['features'],
            target_name=exp_config['target']
        )
        
        # Save validation data
        self._save_dataset(X_val_kj, y_val_kj, metadata_val_kj,
                          exp_path / 'val_korjap_data.csv')
        
        # Process PANGAEA validation
        logger.info("Processing PANGAEA validation data...")
        val_pangaea_df = self.data_loader.load_raw_data('val_pangaea')
        X_val_pg, y_val_pg, metadata_val_pg = self.feature_engineer.prepare_data(
            val_pangaea_df,
            feature_list=exp_config['features'],
            target_name=exp_config['target']
        )
        
        # Save PANGAEA data
        self._save_dataset(X_val_pg, y_val_pg, metadata_val_pg,
                          exp_path / 'val_pangaea_data.csv')
        
        # Save experiment configuration
        config_extended = exp_config.copy()
        config_extended.update({
            'train_samples': len(X_train),
            'train_stations': metadata_train['station_id'].nunique(),
            'val_korjap_samples': len(X_val_kj),
            'val_korjap_stations': metadata_val_kj['station_id'].nunique(),
            'val_pangaea_samples': len(X_val_pg),
            'val_pangaea_stations': metadata_val_pg['station_id'].nunique(),
            'n_features': len(exp_config['features']),
            'created_at': datetime.now().isoformat()
        })
        
        with open(exp_path / 'config.json', 'w') as f:
            json.dump(config_extended, f, indent=2)
        
        # Log summary
        logger.info(f"Experiment data created in {exp_path}")
        logger.info(f"  Training: {config_extended['train_samples']} samples from {config_extended['train_stations']} stations")
        logger.info(f"  Val KorJap: {config_extended['val_korjap_samples']} samples from {config_extended['val_korjap_stations']} stations")
        logger.info(f"  Val PANGAEA: {config_extended['val_pangaea_samples']} samples from {config_extended['val_pangaea_stations']} stations")
        
        return exp_path
    
    def _save_dataset(self, X: pd.DataFrame, y: pd.Series, 
                     metadata: pd.DataFrame, filepath: Path):
        """Save dataset to CSV file"""
        # Combine all data
        full_data = pd.concat([metadata, X], axis=1)
        full_data['target'] = y
        
        # Save to file
        full_data.to_csv(filepath, index=False)
        logger.debug(f"Saved {len(full_data)} samples to {filepath}")
    
    def prepare_all_experiments(self):
        """Prepare data for all experiments in the plan"""
        logger.info(f"Preparing {len(self.experiment_plan)} experiments")
        
        results = []
        for idx, row in self.experiment_plan.iterrows():
            exp_id = int(row['id'])
            try:
                exp_path = self.prepare_experiment_data(exp_id)
                results.append({
                    'id': exp_id,
                    'status': 'success',
                    'path': str(exp_path)
                })
                logger.info(f"Experiment {exp_id} prepared successfully")
            except Exception as e:
                results.append({
                    'id': exp_id,
                    'status': 'failed',
                    'error': str(e)
                })
                logger.error(f"Experiment {exp_id} failed: {e}")
        
        # Save summary
        summary_df = pd.DataFrame(results)
        summary_path = self.experiment_dir / 'preparation_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"Preparation complete. Summary saved to {summary_path}")
        success_count = (summary_df['status'] == 'success').sum()
        logger.info(f"Success: {success_count}/{len(summary_df)}")
        
        return summary_df
    
    def list_experiments(self) -> pd.DataFrame:
        """List all available experiments"""
        return self.experiment_plan[['id', 'features', 'normalization', 'target']].copy()
    
    def get_experiment_info(self, experiment_id: int) -> Dict:
        """Get detailed information about an experiment"""
        exp_config = self.get_experiment(experiment_id)
        
        # Check if data has been prepared
        exp_name = f"exp_{experiment_id:03d}_{exp_config['target']}"
        exp_path = self.experiment_dir / exp_name
        
        if exp_path.exists():
            config_file = exp_path / 'config.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)
                exp_config.update(saved_config)
            
            # Check for existing files
            exp_config['files'] = {
                'train_data': (exp_path / 'train_data.csv').exists(),
                'val_korjap_data': (exp_path / 'val_korjap_data.csv').exists(),
                'val_pangaea_data': (exp_path / 'val_pangaea_data.csv').exists(),
                'config': config_file.exists()
            }
            exp_config['prepared'] = True
        else:
            exp_config['prepared'] = False
        
        return exp_config


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize manager
    manager = ExperimentManager()
    
    # List experiments
    print("\nAvailable experiments:")
    experiments = manager.list_experiments()
    print(experiments.to_string())
    
    # Prepare a specific experiment
    print("\nPreparing experiment 1...")
    try:
        exp_path = manager.prepare_experiment_data(1)
        print(f"Experiment prepared at: {exp_path}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Get experiment info
    print("\nExperiment 1 info:")
    info = manager.get_experiment_info(1)
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Optional: Prepare all experiments
    # print("\nPreparing all experiments...")
    # summary = manager.prepare_all_experiments()
    # print(summary)
