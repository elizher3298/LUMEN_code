#!/usr/bin/env python3
"""
trainer.py - LUMEN Station-based Cross-Validation Training Module
Implements special CV where each fold trains on itself and validates on others
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from datetime import datetime

from .model import ModelConfig, create_compiled_model, get_model_summary

class StationBasedCVTrainer:
    """
    Station-based Cross-Validation Trainer
    Each fold trains on its own data and validates on other folds
    """
    
    def __init__(self, 
                 experiment_id: str,
                 model_config: ModelConfig,
                 output_dir: Path,
                 n_folds: int = 5,
                 cv_threshold: float = 90.0,
                 normalization: str = 'standard',
                 logger: Optional[logging.Logger] = None):
        """
        Initialize trainer
        
        Args:
            experiment_id: Experiment identifier
            model_config: Model configuration
            output_dir: Output directory for results
            n_folds: Number of CV folds (default: 5)
            cv_threshold: CV_RMSE threshold in % (default: 90.0)
            logger: Logger instance
        """
        self.experiment_id = experiment_id
        self.model_config = model_config
        self.output_dir = Path(output_dir)
        self.n_folds = n_folds
        self.cv_threshold = cv_threshold
        self.normalization = normalization
        self.logger = logger or logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Storage for fold results
        self.fold_models: Dict[int, tf.keras.Model] = {}
        self.fold_scalers: Dict[int, StandardScaler] = {}
        self.fold_results: List[Dict[str, Any]] = []
        self.station_fold_mapping: Dict[int, int] = {}
        
        # Best fold tracking
        self.best_fold_idx: int = -1
        self.best_validation_mrmse: float = float('inf')
        
        # Feature tracking
        self.input_features: List[str] = []
        self.target_type: str = ''
        
    def split_stations_to_folds(self, train_data: pd.DataFrame) -> Dict[int, List[int]]:
        """
        Split stations into folds
        
        Args:
            train_data: Training dataframe with station_id column
            
        Returns:
            Dictionary mapping fold index to list of station IDs
        """
        unique_stations = train_data['station_id'].unique()
        n_stations = len(unique_stations)
        
        self.logger.info(f"Splitting {n_stations} stations into {self.n_folds} folds")
        
        # Shuffle stations for random assignment
        np.random.seed(42)  # For reproducibility
        shuffled_stations = np.random.permutation(unique_stations)
        
        # Split stations into folds
        fold_stations = {}
        stations_per_fold = n_stations // self.n_folds
        
        for fold_idx in range(self.n_folds):
            start_idx = fold_idx * stations_per_fold
            if fold_idx == self.n_folds - 1:
                # Last fold gets remaining stations
                fold_stations[fold_idx] = shuffled_stations[start_idx:].tolist()
            else:
                end_idx = start_idx + stations_per_fold
                fold_stations[fold_idx] = shuffled_stations[start_idx:end_idx].tolist()
            
            # Store mapping for each station
            for station_id in fold_stations[fold_idx]:
                self.station_fold_mapping[station_id] = fold_idx
            
            self.logger.info(f"Fold {fold_idx}: {len(fold_stations[fold_idx])} stations")
        
        return fold_stations

    def create_scaler(self) -> Union[StandardScaler, MinMaxScaler]:
        """
        Create scaler based on normalization type
        
        Returns:
            Scaler instance
        """
        if self.normalization == 'minmax':
            return MinMaxScaler(feature_range=(0, 1.2))
        elif self.normalization == 'standard':
            return StandardScaler()
        else:
            self.logger.warning(f"Unknown normalization: {self.normalization}, using StandardScaler")
            return StandardScaler()
    
    def prepare_fold_data(self, 
                          X: np.ndarray, 
                          y: np.ndarray,
                          metadata: pd.DataFrame,
                          fold_idx: int,
                          fold_stations: Dict[int, List[int]]) -> Tuple[np.ndarray, np.ndarray, 
                                                                        np.ndarray, np.ndarray,
                                                                        pd.DataFrame, pd.DataFrame]:
        """
        Prepare training and validation data for a fold
        
        Args:
            X: Feature matrix
            y: Target values
            metadata: Metadata dataframe
            fold_idx: Current fold index
            fold_stations: Station assignments to folds
            
        Returns:
            X_train, y_train, X_val, y_val, metadata_train, metadata_val
        """
        # Get stations for training (current fold only)
        
        # Get stations for validation (all other folds)
        train_stations = []
        for i in range(self.n_folds):
            if i != fold_idx:
                train_stations.extend(fold_stations[i])

        val_stations = fold_stations[fold_idx]
        #train_stations = fold_stations[fold_idx]
        
        # Get stations for validation (all other folds)
        #val_stations = []
        #for i in range(self.n_folds):
        #    if i != fold_idx:
        #        val_stations.extend(fold_stations[i])
        
        # Create masks
        train_mask = metadata['station_id'].isin(train_stations)
        val_mask = metadata['station_id'].isin(val_stations)
        
        # Split data
        X_train = X[train_mask]
        y_train = y[train_mask]
        metadata_train = metadata[train_mask].copy()
        
        X_val = X[val_mask]
        y_val = y[val_mask]
        metadata_val = metadata[val_mask].copy()
        
        self.logger.info(f"Fold {fold_idx}: Train samples={len(X_train)}, Val samples={len(X_val)}")
        
        return X_train, y_train, X_val, y_val, metadata_train, metadata_val


    def train_fold(self,
                   fold_idx: int,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: np.ndarray,
                   y_val: np.ndarray,
                   metadata_train: pd.DataFrame,
                   metadata_val: pd.DataFrame,
                   epochs: int = 100,
                   batch_size: int = 32,
                   patience: int = 20) -> Dict[str, Any]:
        """
        Train a single fold
    
        Args:
            fold_idx: Fold index
            X_train, y_train: Training data
            X_val, y_val: Validation data
            metadata_train, metadata_val: Metadata
            epochs: Maximum epochs
            batch_size: Batch size
            patience: Early stopping patience
        
        Returns:
            Fold results dictionary
        """
        fold_start = time.time()
    
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Training Fold {fold_idx}")
        self.logger.info(f"Normalization: {self.normalization}")
        self.logger.info(f"{'='*60}")
    
        # Normalize features
        scaler = self.create_scaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
    
        # Store scaler
        self.fold_scalers[fold_idx] = scaler
        
        # Create model
        model = create_compiled_model(self.model_config)
    
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
    
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    
        # Store model
        self.fold_models[fold_idx] = model
    
        # Save model
        model_path = self.models_dir / f"fold_{fold_idx}_model.h5"
        model.save(model_path)
    
        # Save scaler
        scaler_path = self.models_dir / f"fold_{fold_idx}_scaler.joblib"
        import joblib
        joblib.dump(scaler, scaler_path)
    
        # Save fold training data info
        fold_data_info = {
            'fold': fold_idx,
            'train_stations': metadata_train['station_id'].unique().tolist(),
            'val_stations': metadata_val['station_id'].unique().tolist(),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'input_features': self.input_features,
            'target_type': self.target_type,
            'normalization': self.normalization
        }
    
        fold_info_path = self.data_dir / f"fold_{fold_idx}_info.json"
        with open(fold_info_path, 'w') as f:
            json.dump(fold_data_info, f, indent=2)
    
        # Calculate training metrics
        train_pred = model.predict(X_train_scaled, verbose=0).flatten()
        val_pred = model.predict(X_val_scaled, verbose=0).flatten()
    
        # ===== 수정된 부분: Target type에 따른 변환 =====
    
        # Target type에 따른 변환
        if self.target_type == 'srad_clearness':
            # 모델 출력은 clearness index (0-1)
            # 일사량으로 변환해야 mRMSE 계산 가능
            train_pred_srad = train_pred * metadata_train['theoretical'].values
            val_pred_srad = val_pred * metadata_val['theoretical'].values
        
            # 메타데이터에 두 가지 모두 저장
            metadata_train['prediction_clearness'] = train_pred  # 원본 clearness 예측
            metadata_train['prediction'] = train_pred_srad       # 변환된 일사량
            metadata_val['prediction_clearness'] = val_pred
            metadata_val['prediction'] = val_pred_srad
        
            # RMSE 계산용 (일사량 단위)
            train_rmse = np.sqrt(np.mean((metadata_train['srad'].values - train_pred_srad) ** 2))
            val_rmse = np.sqrt(np.mean((metadata_val['srad'].values - val_pred_srad) ** 2))
        
        elif self.target_type == 'srad':
            # 모델 출력이 이미 일사량
            metadata_train['prediction'] = train_pred
            metadata_val['prediction'] = val_pred
        
            # RMSE 계산
            train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
            val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
        
        else:
            # 기본값 (이전 버전과의 호환성)
            metadata_train['prediction'] = train_pred
            metadata_val['prediction'] = val_pred
            train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
            val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
            self.logger.warning(f"Unknown target type: {self.target_type}, treating as raw values")
    
        # Error 계산 (이제 같은 단위)
        metadata_train['error'] = metadata_train['srad'].values - metadata_train['prediction'].values
        metadata_val['error'] = metadata_val['srad'].values - metadata_val['prediction'].values
    
        # ===== 수정 끝 =====
    
        fold_result = {
            'fold': fold_idx,
            'training_time': time.time() - fold_start,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'train_stations': metadata_train['station_id'].unique().tolist(),
            'val_stations': metadata_val['station_id'].unique().tolist(),
            'train_rmse': float(train_rmse),
            'val_rmse': float(val_rmse),
            'target_type': self.target_type,
            'epochs_trained': len(history.history['loss']),
            'final_lr': float(tf.keras.backend.get_value(model.optimizer.learning_rate))
        }
    
        self.logger.info(f"Fold {fold_idx} completed - Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
    
        return fold_result, metadata_val
    
    def calculate_station_rmse(self, predictions_df: pd.DataFrame) -> Dict[int, float]:
        """
        Calculate RMSE for each station
        
        Args:
            predictions_df: DataFrame with station_id, srad, and prediction columns
            
        Returns:
            Dictionary mapping station_id to RMSE
        """
        station_rmse = {}
        
        for station_id in predictions_df['station_id'].unique():
            station_data = predictions_df[predictions_df['station_id'] == station_id]
            
            if len(station_data) > 0:
                observed = station_data['srad'].values
                predicted = station_data['prediction'].values
                rmse = np.sqrt(np.mean((observed - predicted) ** 2))
                station_rmse[int(station_id)] = float(rmse)
        
        return station_rmse

    def evaluate_fold_on_other_folds(self,
                                      fold_idx: int,
                                      X: np.ndarray,
                                      y: np.ndarray,
                                      metadata: pd.DataFrame,
                                      fold_stations: Dict[int, List[int]]) -> Dict[str, Any]:
        """
        Evaluate a fold's model on all other folds
        MODIFIED: Calculate CV_RMSE based on individual station RMSEs
        """
        model = self.fold_models[fold_idx]
        scaler = self.fold_scalers[fold_idx]
    
        # 모든 station의 RMSE를 개별적으로 저장
        all_station_rmse = []
        fold_wise_results = {}
    
        for other_fold_idx in range(self.n_folds):
            if other_fold_idx == fold_idx:
                continue  # Skip self
        
            # Get data for other fold
            other_fold_stations = fold_stations[other_fold_idx]
            mask = metadata['station_id'].isin(other_fold_stations)
        
            X_other = X[mask]
            y_other = y[mask]
            metadata_other = metadata[mask].copy()
        
            # Scale and predict
            X_other_scaled = scaler.transform(X_other)
            predictions = model.predict(X_other_scaled, verbose=0).flatten()
        
            if self.target_type == 'srad_clearness':
                # 모델 출력은 clearness index, srad로 변환
                predictions_srad = predictions * metadata_other['theoretical'].values
                metadata_other['prediction_clearness'] = predictions
                metadata_other['prediction'] = predictions_srad  # srad 저장
            elif self.target_type == 'srad':
                # 모델 출력이 이미 srad
                metadata_other['prediction'] = predictions
            else:
                # Handle clearness type
                if self.target_type == "clearness":
                    # Convert clearness to srad for RMSE calculation
                    predictions_srad = predictions * metadata_other["theoretical"].values
                    metadata_other["prediction_clearness"] = predictions
                    metadata_other["prediction"] = predictions_srad  # Store as srad
                else:
                    # Handle clearness type
                    if self.target_type == "clearness":
                        # Convert clearness to srad for RMSE calculation
                        predictions_srad = predictions * metadata_other["theoretical"].values
                        metadata_other["prediction_clearness"] = predictions
                        metadata_other["prediction"] = predictions_srad  # Store as srad
                    else:
                        # Unknown type - use raw predictions
                        metadata_other["prediction"] = predictions
                        self.logger.warning(f"Unknown target type in evaluation: {self.target_type}")
            # Calculate station-wise RMSE
            station_rmse = self.calculate_station_rmse(metadata_other)
        
            # 각 station의 RMSE를 전체 리스트에 추가
            all_station_rmse.extend(station_rmse.values())
        
            # Calculate mRMSE (mean of station RMSEs) 
            mrmse = np.mean(list(station_rmse.values()))
        
            fold_wise_results[f'fold_{other_fold_idx}'] = {
                'mRMSE': float(mrmse),
                'station_rmse': station_rmse,
                'n_stations': len(station_rmse),
                'n_samples': len(X_other)
            }
    
        # Station 레벨 CV_RMSE (새로운 방식)
        all_station_rmse = np.array(all_station_rmse)
        station_mean_rmse = np.mean(all_station_rmse)
        station_std_rmse = np.std(all_station_rmse)
        station_cv_rmse = (station_std_rmse / station_mean_rmse) * 100 if station_mean_rmse > 0 else float('inf')
    
        # 기존 Fold 레벨 계산도 유지 (호환성)
        other_folds_mrmse = [fold_wise_results[f'fold_{i}']['mRMSE'] 
                             for i in range(self.n_folds) if i != fold_idx]
        validation_mrmse = np.mean(other_folds_mrmse)
        std_mrmse = np.std(other_folds_mrmse)
        cv_rmse = (std_mrmse / validation_mrmse) * 100 if validation_mrmse > 0 else float('inf')
    
        return {
            'fold': fold_idx,
            # 기존 방식 (호환성 유지)
            'validation_mRMSE': float(validation_mrmse),
            'CV_RMSE': float(station_cv_rmse),  # ← Station CV_RMSE 사용!
            'std_mRMSE': float(station_std_rmse),  # ← Station std 사용!
            'other_folds_mRMSE': [float(x) for x in other_folds_mrmse],
            'fold_wise_results': fold_wise_results,
            # 추가 정보
            'cv_metrics': {
                'station_level': {
                    'mean_rmse': float(station_mean_rmse),
                    'std_rmse': float(station_std_rmse),
                    'cv_rmse': float(station_cv_rmse),
                    'n_stations': len(all_station_rmse)
                },
                'fold_level': {
                    'mean_mrmse': float(validation_mrmse),
                    'std_mrmse': float(std_mrmse),
                    'cv_rmse': float(cv_rmse)
                }
            }
        }

  
    def run_cross_validation(self,
                             X: np.ndarray,
                             y: np.ndarray,
                             metadata: pd.DataFrame,
                             features: List[str] = None,
                             target_type: str = 'srad_clearness',
                             epochs: int = 100,
                             batch_size: int = 32,
                             patience: int = 20) -> Dict[str, Any]:
        """
        Run complete cross-validation
        
        Args:
            X: Feature matrix
            y: Target values (clearness index)
            metadata: Metadata with station_id, date, srad, theoretical
            features: List of feature names
            target_type: Type of target ('srad_clearness' or 'srad')
            epochs: Maximum epochs
            batch_size: Batch size
            patience: Early stopping patience
            
        Returns:
            Cross-validation results
        """
        cv_start = time.time()
        
        # Store features and target info
        self.input_features = features if features else [f'feature_{i}' for i in range(X.shape[1])]
        self.target_type = target_type
        
        # Save input/output data
        self.save_experiment_data(X, y, metadata)
        
        # Split stations into folds
        fold_stations = self.split_stations_to_folds(metadata)
        
        # Train each fold
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting {self.n_folds}-Fold Cross-Validation")
        self.logger.info(f"{'='*60}")
        
        for fold_idx in range(self.n_folds):
            # Prepare fold data
            X_train, y_train, X_val, y_val, metadata_train, metadata_val = self.prepare_fold_data(
                X, y, metadata, fold_idx, fold_stations
            )
            
            # Train fold
            fold_result, val_predictions = self.train_fold(
                fold_idx, X_train, y_train, X_val, y_val,
                metadata_train, metadata_val,
                epochs, batch_size, patience
            )
            
            # Store initial results
            self.fold_results.append(fold_result)
        
        # Evaluate each fold on other folds
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Evaluating each fold on other folds")
        self.logger.info(f"{'='*60}")
        
        for fold_idx in range(self.n_folds):
            eval_result = self.evaluate_fold_on_other_folds(
                fold_idx, X, y, metadata, fold_stations
            )
            
            # Add evaluation results to fold results
            self.fold_results[fold_idx].update(eval_result)
            
            self.logger.info(
                f"Fold {fold_idx}: validation_mRMSE={eval_result['validation_mRMSE']:.4f}, "
                f"CV_RMSE={eval_result['CV_RMSE']:.2f}%"
            )
        
        # Select best fold
        self.select_best_fold()
        
        # Prepare final results
        cv_results = {
            'experiment_id': self.experiment_id,
            'n_folds': self.n_folds,
            'cv_threshold': self.cv_threshold,
            'input_features': self.input_features,
            'target_type': self.target_type,
            'fold_results': self.fold_results,
            'best_fold': self.best_fold_idx,
            'best_validation_mRMSE': float(self.best_validation_mrmse),
            'station_fold_mapping': {int(k): int(v) for k, v in self.station_fold_mapping.items()},
            'training_time': time.time() - cv_start,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        self.save_cv_results(cv_results)
        
        return cv_results
    
    def select_best_fold(self):
        """Select best fold based on CV_RMSE threshold
        MODIFIED: Shows both station-level and fold-level metrics
        """
        valid_folds = []
    
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Selecting best fold (CV_RMSE threshold: {self.cv_threshold}%)")
        self.logger.info(f"{'='*60}")
    
        for fold_idx, result in enumerate(self.fold_results):
            cv_rmse = result.get('CV_RMSE', float('inf'))  # Now using station-level
            validation_mrmse = result.get('validation_mRMSE', float('inf'))
        
            # Get both metrics if available
            if 'cv_metrics' in result:
                station_cv = result['cv_metrics']['station_level']['cv_rmse']
                fold_cv = result['cv_metrics']['fold_level']['cv_rmse']
                self.logger.info(
                    f"Fold {fold_idx}: Station CV_RMSE={station_cv:.2f}%, "
                    f"Fold CV_RMSE={fold_cv:.2f}%, mRMSE={validation_mrmse:.4f}"
                )
            else:
                self.logger.info(
                    f"Fold {fold_idx}: CV_RMSE={cv_rmse:.2f}%, "
                    f"validation_mRMSE={validation_mrmse:.4f}"
                )
        
            if cv_rmse <= self.cv_threshold:
                valid_folds.append((fold_idx, validation_mrmse))
                self.logger.info(f"  ✓ Meets CV_RMSE threshold")
            else:
                self.logger.info(f"  ✗ Exceeds CV_RMSE threshold")
    
        if valid_folds:
            # Select fold with lowest validation_mRMSE
            self.best_fold_idx, self.best_validation_mrmse = min(valid_folds, key=lambda x: x[1])
            self.logger.info(
                f"\n✓ Best fold: {self.best_fold_idx} with "
                f"validation_mRMSE={self.best_validation_mrmse:.4f}"
            )
        else:
            self.best_fold_idx = -1
            self.best_validation_mrmse = float('inf')
            self.logger.warning("\n✗ No fold meets CV_RMSE threshold - model rejected")
    
    def save_cv_results(self, results: Dict[str, Any]):
        """Save cross-validation results"""
        results_path = self.output_dir / "cv_results.json"
        
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj

        results['normalization'] = self.normalization

        results = convert_types(results)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"CV results saved to {results_path}")
        
        # Save summary
        if self.best_fold_idx >= 0:
            summary = {
                'experiment_id': self.experiment_id,
                'status': 'accepted',
                'best_fold': self.best_fold_idx,
                'best_validation_mRMSE': self.best_validation_mrmse,
                'best_CV_RMSE': results['fold_results'][self.best_fold_idx]['CV_RMSE'],
                'normalization': self.normalization
            }
        else:
            summary = {
                'experiment_id': self.experiment_id,
                'status': 'rejected',
                'reason': 'No fold meets CV_RMSE threshold',
                'normalization': self.normalization
            }
        
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary saved to {summary_path}")
    
    def get_best_model(self) -> Optional[Tuple[tf.keras.Model, StandardScaler]]:
        """
        Get best model and its scaler
        
        Returns:
            Tuple of (model, scaler) if available, None otherwise
        """
        if self.best_fold_idx >= 0:
            return (
                self.fold_models[self.best_fold_idx],
                self.fold_scalers[self.best_fold_idx]
            )
        return None
    
    def save_experiment_data(self, X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame):
        """Save full training data and feature information"""
        
        # Save feature names and descriptions
        feature_info = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'input_features': self.input_features,
            'n_features': len(self.input_features),
            'target_type': self.target_type,
            'target_description': {
                'srad_clearness': 'Clearness index (srad/theoretical)',
                'srad': 'Solar radiation (MJ/m²/day)'
            }.get(self.target_type, self.target_type),
            'n_samples': len(X),
            'n_stations': metadata['station_id'].nunique(),
            'date_range': {
                'min': str(metadata['date'].min()),
                'max': str(metadata['date'].max())
            }
        }
        
        feature_path = self.data_dir / "feature_info.json"
        with open(feature_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        self.logger.info(f"Feature information saved to {feature_path}")
        
        # Save FULL training data (not sample)
        full_data_df = pd.DataFrame(X, columns=self.input_features)
        full_data_df['target'] = y
        full_data_df['station_id'] = metadata['station_id'].values
        full_data_df['date'] = metadata['date'].values
        
        # Add original srad and theoretical for reference
        if 'srad' in metadata.columns:
            full_data_df['srad_original'] = metadata['srad'].values
        if 'theoretical' in metadata.columns:
            full_data_df['theoretical'] = metadata['theoretical'].values
        if 'himawari_srad' in metadata.columns:
            full_data_df['himawari_srad'] = metadata['himawari_srad'].values
        if 'agera5_srad' in metadata.columns:
            full_data_df['agera5_srad'] = metadata['agera5_srad'].values
        
        # Save full training data
        train_data_path = self.data_dir / "training_data_full.csv"
        full_data_df.to_csv(train_data_path, index=False)
        
        self.logger.info(f"Full training data ({len(full_data_df)} rows) saved to {train_data_path}")
        
        # Save data shape and basic info
        data_info = {
            'full_shape': {
                'X': list(X.shape),
                'y': list(y.shape)
            },
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_stations': metadata['station_id'].nunique(),
            'station_list': sorted(metadata['station_id'].unique().tolist()),
            'date_range': {
                'min': str(metadata['date'].min()),
                'max': str(metadata['date'].max())
            },
            'file_size_mb': train_data_path.stat().st_size / (1024 * 1024)
        }
        
        data_info_path = self.data_dir / "data_info.json"
        with open(data_info_path, 'w') as f:
            json.dump(data_info, f, indent=2)
        
        self.logger.info(f"Data information saved to {data_info_path}")
