#!/usr/bin/env python3
"""
validator.py - LUMEN Independent Validation Module
Validates best model on Korea/Japan (2016-2020) and PANGAEA datasets
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

class ModelValidator:
    """
    Validator for trained models on independent datasets
    """
    
    def __init__(self,
                 experiment_id: str,
                 model: tf.keras.Model,
                 scaler: StandardScaler,
                 output_dir: Path,
                 features: List[str] = None,
                 target_type: str = 'srad_clearness',
                 logger: Optional[logging.Logger] = None):
        """
        Initialize validator
        
        Args:
            experiment_id: Experiment identifier
            model: Trained TensorFlow model
            scaler: Fitted StandardScaler
            output_dir: Output directory
            features: List of feature names
            target_type: Type of target ('srad_clearness' or 'srad')
            logger: Logger instance
        """
        self.experiment_id = experiment_id
        self.model = model
        self.scaler = scaler
        self.output_dir = Path(output_dir)
        self.features = features or []
        self.target_type = target_type
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = self.output_dir / "predictions"
        self.predictions_dir.mkdir(exist_ok=True)
        
    def calculate_metrics(self, 
                          observed: np.ndarray, 
                          predicted: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Args:
            observed: Observed values
            predicted: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(observed) | np.isnan(predicted))
        observed = observed[mask]
        predicted = predicted[mask]
        
        if len(observed) == 0:
            return {
                'rmse': float('nan'),
                'mae': float('nan'),
                'mbe': float('nan'),
                'r2': float('nan'),
                'n_samples': 0
            }
        
        # Calculate metrics
        errors = predicted - observed
        
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))
        mbe = np.mean(errors)
        
        # R-squared
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'mbe': float(mbe),
            'r2': float(r2),
            'n_samples': len(observed)
        }
    
    def calculate_station_metrics(self,
                                   predictions_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """
        Calculate metrics for each station
        
        Args:
            predictions_df: DataFrame with station_id, srad, estimate columns
            
        Returns:
            Dictionary mapping station_id to metrics
        """
        station_metrics = {}
        
        for station_id in predictions_df['station_id'].unique():
            station_data = predictions_df[predictions_df['station_id'] == station_id]
            
            if len(station_data) > 0:
                metrics = self.calculate_metrics(
                    station_data['srad'].values,
                    station_data['estimate'].values
                )
                metrics['station_id'] = int(station_id)
                station_metrics[int(station_id)] = metrics
        
        return station_metrics
    
    def calculate_hemisphere_metrics(self,
                                      predictions_df: pd.DataFrame,
                                      station_info: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics by hemisphere (Northern/Southern)
        
        Args:
            predictions_df: Predictions DataFrame
            station_info: Station information with lat column
            
        Returns:
            Dictionary with 'northern' and 'southern' metrics
        """
        hemisphere_metrics = {}
        
        if station_info is None or 'lat' not in station_info.columns:
            self.logger.warning("Station latitude information not available for hemisphere analysis")
            return hemisphere_metrics
        
        # Merge with station info
        merged = predictions_df.merge(
            station_info[['station_id', 'lat']], 
            on='station_id', 
            how='left'
        )
        
        # Northern hemisphere (lat >= 0)
        northern_data = merged[merged['lat'] >= 0]
        if len(northern_data) > 0:
            hemisphere_metrics['northern'] = self.calculate_metrics(
                northern_data['srad'].values,
                northern_data['estimate'].values
            )
            hemisphere_metrics['northern']['n_stations'] = northern_data['station_id'].nunique()
        
        # Southern hemisphere (lat < 0)
        southern_data = merged[merged['lat'] < 0]
        if len(southern_data) > 0:
            hemisphere_metrics['southern'] = self.calculate_metrics(
                southern_data['srad'].values,
                southern_data['estimate'].values
            )
            hemisphere_metrics['southern']['n_stations'] = southern_data['station_id'].nunique()
        
        return hemisphere_metrics
    
    def validate_dataset(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         metadata: pd.DataFrame,
                         dataset_name: str,
                         station_info: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Validate on a dataset
        
        Args:
            X: Feature matrix
            y: Target values (clearness index)
            metadata: Metadata with station_id, date, srad, theoretical
            dataset_name: Name of the dataset (e.g., 'korea_japan', 'pangaea')
            station_info: Optional station information for hemisphere analysis
            
        Returns:
            Validation results
        """
        val_start = datetime.now()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Validating on {dataset_name}")
        self.logger.info(f"{'='*60}")
        
        # Remove samples with NaN
        nan_mask = np.isnan(X).any(axis=1) | np.isnan(y)
        X_clean = X[~nan_mask]
        y_clean = y[~nan_mask]
        metadata_clean = metadata[~nan_mask].copy()
        
        n_removed = nan_mask.sum()
        if n_removed > 0:
            self.logger.info(f"Removed {n_removed} samples with NaN values")
        
        # Scale features
        X_scaled = self.scaler.transform(X_clean)
        
        # Predict clearness index
        clearness_pred = self.model.predict(X_scaled, verbose=0).flatten()
        
        # Convert to radiation
        theoretical = metadata_clean['theoretical'].values
        radiation_pred = clearness_pred * theoretical
        radiation_obs = metadata_clean['srad'].values
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'station_id': metadata_clean['station_id'].values,
            'date': metadata_clean['date'].values,
            'srad': radiation_obs,
            'estimate': radiation_pred,
            'clearness_obs': y_clean,
            'clearness_pred': clearness_pred,
            'theoretical': theoretical
        })
        
        # Add satellite estimates if available
        if 'himawari_srad' in metadata_clean.columns:
            predictions_df['himawari_srad'] = metadata_clean['himawari_srad'].values
        if 'agera5_srad' in metadata_clean.columns:
            predictions_df['agera5_srad'] = metadata_clean['agera5_srad'].values
        
        # Calculate overall metrics
        overall_metrics = self.calculate_metrics(radiation_obs, radiation_pred)
        
        # Calculate station-wise metrics
        station_metrics = self.calculate_station_metrics(predictions_df)
        
        # Calculate mean of station RMSEs
        station_rmse_values = [m['rmse'] for m in station_metrics.values() if not np.isnan(m['rmse'])]
        mrmse = np.mean(station_rmse_values) if station_rmse_values else float('nan')
        
        # Calculate hemisphere metrics
        hemisphere_metrics = self.calculate_hemisphere_metrics(predictions_df, station_info)
        
        # Save predictions (FULL DATA)
        predictions_path = self.predictions_dir / f"{dataset_name}_predictions.csv"
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(predictions_path, index=False)
        self.logger.info(f"Saved predictions to {predictions_path}")
        
        # Save validation input data (FULL DATA)
        val_input_df = pd.DataFrame(X_clean, columns=self.features)
        val_input_df['target'] = y_clean
        val_input_df['station_id'] = metadata_clean['station_id'].values
        val_input_df['date'] = metadata_clean['date'].values
        val_input_df['srad_original'] = metadata_clean['srad'].values
        val_input_df['theoretical'] = metadata_clean['theoretical'].values
        
        # Add satellite data if available
        if 'himawari_srad' in metadata_clean.columns:
            val_input_df['himawari_srad'] = metadata_clean['himawari_srad'].values
        if 'agera5_srad' in metadata_clean.columns:
            val_input_df['agera5_srad'] = metadata_clean['agera5_srad'].values
        
        val_input_path = self.predictions_dir / f"{dataset_name}_input_full.csv"
        val_input_df.to_csv(val_input_path, index=False)
        self.logger.info(f"Saved full validation input data to {val_input_path}")
        
        # Save validation info
        val_info = {
            'dataset': dataset_name,
            'experiment_id': self.experiment_id,
            'input_features': self.features,
            'target_type': self.target_type,
            'n_samples_total': len(metadata),
            'n_samples_used': len(predictions_df),
            'n_samples_removed_nan': n_removed if n_removed > 0 else 0,
            'n_stations': predictions_df['station_id'].nunique(),
            'date_range': {
                'min': str(predictions_df['date'].min()),
                'max': str(predictions_df['date'].max())
            },
            'file_sizes_mb': {
                'predictions': predictions_path.stat().st_size / (1024 * 1024),
                'input_full': val_input_path.stat().st_size / (1024 * 1024)
            }
        }
        
        val_info_path = self.predictions_dir / f"{dataset_name}_info.json"
        with open(val_info_path, 'w') as f:
            json.dump(val_info, f, indent=2)
        self.logger.info(f"Saved validation info to {val_info_path}")
        
        # Log results
        self.logger.info(f"\n{dataset_name} Results:")
        self.logger.info(f"  Overall RMSE: {overall_metrics['rmse']:.4f}")
        self.logger.info(f"  Overall MAE: {overall_metrics['mae']:.4f}")
        self.logger.info(f"  Overall MBE: {overall_metrics['mbe']:.4f}")
        self.logger.info(f"  Overall R²: {overall_metrics['r2']:.4f}")
        self.logger.info(f"  Station mRMSE: {mrmse:.4f}")
        self.logger.info(f"  Number of stations: {len(station_metrics)}")
        self.logger.info(f"  Number of samples: {overall_metrics['n_samples']}")
        
        if hemisphere_metrics:
            for hemi, metrics in hemisphere_metrics.items():
                self.logger.info(f"\n  {hemi.capitalize()} Hemisphere:")
                self.logger.info(f"    RMSE: {metrics['rmse']:.4f}")
                self.logger.info(f"    R²: {metrics['r2']:.4f}")
                self.logger.info(f"    Stations: {metrics.get('n_stations', 'N/A')}")
        
        # Prepare results
        results = {
            'dataset': dataset_name,
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'overall_metrics': overall_metrics,
            'station_mRMSE': float(mrmse),
            'n_stations': len(station_metrics),
            'station_metrics': station_metrics,
            'hemisphere_metrics': hemisphere_metrics,
            'validation_time': (datetime.now() - val_start).total_seconds(),
            'predictions_file': str(predictions_path)
        }
        
        return results
    
    def validate_korea_japan(self,
                             val_data_path: Path,
                             features: List[str]) -> Dict[str, Any]:
        """
        Validate on Korea/Japan 2016-2020 dataset
        
        Args:
            val_data_path: Path to validation data CSV
            features: List of feature names
            
        Returns:
            Validation results
        """
        # Load data
        val_data = pd.read_csv(val_data_path)
        
        # Prepare features
        X = val_data[features].values
        
        # Calculate clearness index as target
        y = val_data['srad'].values / val_data['theoretical'].values
        
        # Prepare metadata
        metadata = val_data[['station_id', 'date', 'srad', 'theoretical']].copy()
        
        # Add satellite data if available
        if 'himawari_srad' in val_data.columns:
            metadata['himawari_srad'] = val_data['himawari_srad']
        if 'agera5_srad' in val_data.columns:
            metadata['agera5_srad'] = val_data['agera5_srad']
        
        # Get station info if available
        station_info = None
        if 'lat' in val_data.columns:
            station_info = val_data[['station_id', 'lat']].drop_duplicates()
        
        return self.validate_dataset(X, y, metadata, 'korea_japan', station_info)
    
    def validate_pangaea(self,
                         val_data_path: Path,
                         features: List[str],
                         pangaea_info_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Validate on PANGAEA dataset
        
        Args:
            val_data_path: Path to validation data CSV
            features: List of feature names
            pangaea_info_path: Optional path to station info CSV
            
        Returns:
            Validation results
        """
        # Load data
        val_data = pd.read_csv(val_data_path)
        
        # Prepare features
        X = val_data[features].values
        
        # Calculate clearness index as target
        y = val_data['srad'].values / val_data['theoretical'].values
        
        # Prepare metadata
        metadata = val_data[['station_id', 'date', 'srad', 'theoretical']].copy()
        
        # Add satellite data if available
        if 'himawari_srad' in val_data.columns:
            metadata['himawari_srad'] = val_data['himawari_srad']
        if 'agera5_srad' in val_data.columns:
            metadata['agera5_srad'] = val_data['agera5_srad']
        
        # Load station info if provided
        station_info = None
        if pangaea_info_path and pangaea_info_path.exists():
            station_info = pd.read_csv(pangaea_info_path)
            # Rename columns if necessary
            if 'stn' in station_info.columns:
                station_info = station_info.rename(columns={'stn': 'station_id'})
        elif 'lat' in val_data.columns:
            station_info = val_data[['station_id', 'lat']].drop_duplicates()
        
        return self.validate_dataset(X, y, metadata, 'pangaea', station_info)
    
    def run_all_validations(self,
                            val_korjap_path: Optional[Path] = None,
                            val_pangaea_path: Optional[Path] = None,
                            features: List[str] = None,
                            pangaea_info_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run all available validations
        
        Args:
            val_korjap_path: Path to Korea/Japan validation data
            val_pangaea_path: Path to PANGAEA validation data
            features: List of feature names
            pangaea_info_path: Optional path to PANGAEA station info
            
        Returns:
            Combined validation results
        """
        all_results = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'input_features': self.features,
            'target_type': self.target_type,
            'validations': {}
        }
        
        # Validate Korea/Japan if available
        if val_korjap_path and val_korjap_path.exists() and features:
            self.logger.info("Running Korea/Japan validation...")
            korjap_results = self.validate_korea_japan(val_korjap_path, features)
            all_results['validations']['korea_japan'] = korjap_results
        
        # Validate PANGAEA if available
        if val_pangaea_path and val_pangaea_path.exists() and features:
            self.logger.info("Running PANGAEA validation...")
            pangaea_results = self.validate_pangaea(
                val_pangaea_path, features, pangaea_info_path
            )
            all_results['validations']['pangaea'] = pangaea_results
        
        # Save combined results
        results_path = self.output_dir / "validation_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        self.logger.info(f"\nValidation results saved to {results_path}")
        
        # Create summary
        summary = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat()
        }
        
        for name, results in all_results['validations'].items():
            summary[name] = {
                'rmse': results['overall_metrics']['rmse'],
                'mae': results['overall_metrics']['mae'],
                'r2': results['overall_metrics']['r2'],
                'mrmse': results['station_mRMSE'],
                'n_stations': results['n_stations'],
                'n_samples': results['overall_metrics']['n_samples']
            }
        
        summary_path = self.output_dir / "validation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Validation summary saved to {summary_path}")
        
        return all_results
