#!/usr/bin/env python3
"""
main.py - LUMEN Main Execution Pipeline
Auto-iterates all architecture x dropout combinations from ModelArchitectures
Usage:
  python main.py exp5                          # Run all combinations
  python main.py exp5 --architecture double_16_16 --dropout-rate 0.1  # Run single
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
import traceback
from typing import Optional, Any, Dict, List

sys.path.append(str(Path(__file__).parent))

from src.model import ModelConfig, ModelArchitectures
from src.trainer import StationBasedCVTrainer
from src.validator import ModelValidator

# Repository root (directory containing this file).
BASE_DIR = Path(__file__).resolve().parent


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"
    
    # Reset root logger handlers to avoid duplicate logs across combinations
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('LUMEN')
    logger.info(f"Logging to {log_file}")
    return logger


def get_experiment_config(experiment_id: str, base_dir: Path) -> Dict:
    """
    Get experiment configuration from CSV file
    
    Args:
        experiment_id: Experiment ID (e.g., 'exp5' or 'exp9')
        base_dir: Base directory
        
    Returns:
        Dictionary with experiment configuration including normalization
    """
    csv_path = base_dir / "configs" / "gpu_experiment_plan.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Experiment plan not found: {csv_path}")
    
    plan_df = pd.read_csv(csv_path)
    
    # Find row with matching ID (string comparison)
    exp_row = plan_df[plan_df['id'] == experiment_id]
    
    if exp_row.empty:
        # Try with 'exp' prefix if not found
        if not experiment_id.startswith('exp'):
            exp_row = plan_df[plan_df['id'] == f'exp{experiment_id}']
    
    if exp_row.empty:
        raise ValueError(f"Experiment ID {experiment_id} not found in CSV")
    
    exp = exp_row.iloc[0]
    feature_list = exp['features'].split()
    
    config = {
        'id': exp['id'],
        'features': feature_list,
        'target': exp.get('target', 'srad'),
        'normalization': exp.get('normalization', 'standard')
    }
    
    return config


def load_experiment_data(experiment_id: str, base_dir: Path, logger: logging.Logger, 
                        target_from_csv: str = None) -> dict:
    """
    Load pre-processed data from experiments/ folder
    
    Args:
        experiment_id: Experiment ID
        base_dir: Base directory
        logger: Logger
        target_from_csv: Target type from CSV configuration
        
    Returns:
        dict with keys: X_train, y_train, metadata_train, features, target_type, 
                       val_korjap_path, val_pangaea_path
    """
    exp_dir = base_dir / "experiments" / experiment_id
    
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    
    # Load training data
    train_path = exp_dir / "train_data.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded training data: {train_df.shape}")
    
    # Metadata columns
    metadata_cols = ['station_id', 'date', 'srad_original', 'theoretical']
    if 'srad' not in train_df.columns and 'srad_original' in train_df.columns:
        train_df['srad'] = train_df['srad_original']
    
    # Separate features and target
    exclude_cols = metadata_cols + ['station_name', 'lat', 'lon', 'alt', 
                                    'dist_to_coast_km', 'target', 'srad']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    
    # Metadata
    metadata_train = train_df[['station_id', 'date', 'srad', 'theoretical']].copy()
    
    # Validation data paths
    val_korjap_path = exp_dir / "val_korjap_data.csv"
    val_pangaea_path = exp_dir / "val_pangaea_data.csv"
    
    # Set target type - prefer CSV value
    if target_from_csv:
        target_type = target_from_csv
        logger.info(f"Using target type from CSV: {target_type}")
    else:
        # Fallback: infer from target value range
        if 'target' in train_df.columns:
            if train_df['target'].max() > 2:
                target_type = 'srad'
            else:
                target_type = 'srad_clearness'
        else:
            target_type = 'srad_clearness'
        logger.info(f"Inferred target type: {target_type}")
    
    logger.info(f"Features ({len(feature_cols)}): {feature_cols}")
    logger.info(f"Target type: {target_type}")
    logger.info(f"Training samples: {len(X_train)}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'metadata_train': metadata_train,
        'features': feature_cols,
        'target_type': target_type,
        'val_korjap_path': val_korjap_path,
        'val_pangaea_path': val_pangaea_path
    }


def run_experiment(experiment_id: str,
                   base_dir: Path = BASE_DIR,
                   output_dir: Optional[Path] = None,
                   architecture: Optional[str] = None,
                   hidden_layers: Optional[List[int]] = None,
                   dropout_rate: float = 0.2,
                   combination_name: Optional[str] = None,
                   use_batch_norm: bool = False,
                   epochs: int = 100,
                   batch_size: int = 32,
                   patience: int = 20,
                   cv_threshold: float = 90.0):
    """
    Run a single experiment with specified architecture and dropout
    
    Args:
        experiment_id: Experiment ID (e.g., 'exp2')
        base_dir: Base directory
        output_dir: Output directory (if None, uses base_dir/results)
        architecture: Model architecture name (used to look up hidden_layers)
        hidden_layers: Direct hidden layer specification (overrides architecture lookup)
        dropout_rate: Dropout rate
        combination_name: Name for this combination (used for subfolder)
        use_batch_norm: Use batch normalization
        epochs: Maximum epochs
        batch_size: Batch size
        patience: Early stopping patience
        cv_threshold: CV_RMSE threshold
    """
    # Set output directory
    if output_dir is None:
        output_dir = base_dir / "results" / experiment_id
    else:
        output_dir = Path(output_dir)
    
    # Create subfolder for this combination
    if combination_name:
        output_dir = output_dir / combination_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    try:
        logger.info("=" * 60)
        logger.info(f"Starting Experiment: {experiment_id}")
        if combination_name:
            logger.info(f"Combination: {combination_name}")
        logger.info("=" * 60)
        logger.info(f"Output directory: {output_dir}")
        
        # Get experiment config from CSV
        exp_config = get_experiment_config(experiment_id, base_dir)
        normalization = exp_config.get('normalization', 'standard')
        target_from_csv = exp_config.get('target', 'srad')
        
        logger.info(f"Experiment configuration from CSV:")
        logger.info(f"  - Features: {exp_config.get('features', [])}")
        logger.info(f"  - Target: {target_from_csv}")
        logger.info(f"  - Normalization: {normalization}")
        
        # Load experiment data
        data = load_experiment_data(experiment_id, base_dir, logger, target_from_csv)
        
        # Determine model architecture
        if hidden_layers:
            # Directly specified (from get_all_combinations)
            pass
        elif architecture:
            hidden_layers = ModelArchitectures.get_architecture(architecture)
        else:
            raise ValueError("Either --architecture or hidden_layers must be specified")
        
        logger.info(f"Model architecture: {hidden_layers}")
        logger.info(f"Dropout rate: {dropout_rate}")
        
        # Save config
        config_save = {
            'experiment_id': experiment_id,
            'combination_name': combination_name,
            'csv_config': {
                'features': exp_config.get('features', []),
                'target': target_from_csv,
                'normalization': normalization
            },
            'model': {
                'architecture': architecture or combination_name or 'custom',
                'hidden_layers': hidden_layers,
                'dropout_rate': dropout_rate,
                'use_batch_norm': use_batch_norm
            },
            'training': {
                'epochs': epochs,
                'batch_size': batch_size,
                'patience': patience,
                'cv_threshold': cv_threshold
            },
            'data': {
                'features': data['features'],
                'target_type': data['target_type'],
                'n_samples': len(data['X_train']),
                'normalization_used': normalization
            },
            'output_directory': str(output_dir)
        }
        
        config_path = output_dir / "experiment_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_save, f, indent=2)
        
        # Model config
        model_config = ModelConfig(
            input_size=len(data['features']),
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        
        # Initialize trainer
        trainer = StationBasedCVTrainer(
            experiment_id=experiment_id,
            model_config=model_config,
            output_dir=output_dir,
            n_folds=5,
            normalization=normalization,
            cv_threshold=cv_threshold,
            logger=logger
        )
        
        logger.info(f"Trainer initialized with normalization: {normalization}")
        
        # Run cross-validation
        logger.info("\n" + "=" * 60)
        logger.info("Starting Cross-Validation Training")
        logger.info("=" * 60)
        
        cv_results = trainer.run_cross_validation(
            data['X_train'], 
            data['y_train'], 
            data['metadata_train'],
            features=data['features'],
            target_type=data['target_type'],
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        
        # Print results summary
        logger.info("\n" + "=" * 60)
        logger.info("Cross-Validation Results Summary")
        logger.info("=" * 60)
        logger.info(f"Best Fold: {cv_results['best_fold']}")
        logger.info(f"Best Validation mRMSE: {cv_results['best_validation_mRMSE']:.4f} MJ/m²/day")
        
        # Per-fold results
        logger.info("\nFold Results:")
        for fold_idx in range(5):
            if fold_idx in cv_results.get('fold_results', []):
                fold_result = cv_results['fold_results'][fold_idx]
                logger.info(f"  Fold {fold_idx}:")
                logger.info(f"    Training RMSE: {fold_result.get('train_rmse', 'N/A')}")
                logger.info(f"    Validation mRMSE: {fold_result.get('validation_mRMSE', 'N/A')}")
                logger.info(f"    CV_RMSE: {fold_result.get('CV_RMSE', 'N/A'):.2f}%")
        
        # Independent validation (optional)
        if data['val_korjap_path'].exists() or data['val_pangaea_path'].exists():
            logger.info("\n" + "=" * 60)
            logger.info("Running Validation on Independent Datasets")
            logger.info("=" * 60)
            
            best_fold = cv_results.get('best_fold', -1)
            if best_fold < 0:
                logger.warning("No valid model available for validation (all folds rejected)")
                return cv_results
            
            validator = ModelValidator(
                experiment_id=experiment_id,
                model=trainer.fold_models[best_fold],
                scaler=trainer.fold_scalers[best_fold],
                features=data['features'],
                target_type=data['target_type'],
                output_dir=output_dir,
                logger=logger
            )
            
            # Korea/Japan 2016-2020 validation
            if data['val_korjap_path'].exists():
                logger.info("\nValidating on Korea/Japan 2016-2020...")
                val_df = pd.read_csv(data['val_korjap_path'])
                
                X_val = val_df[data['features']].values
                y_val = val_df['target'].values if 'target' in val_df else val_df['srad'].values
                metadata_val = val_df[['station_id', 'date', 'srad', 'theoretical']].copy()
                
                station_info_path = base_dir / "data" / "station_info.csv"
                if station_info_path.exists():
                    station_info = pd.read_csv(station_info_path)
                else:
                    station_info = None
                
                korjap_results = validator.validate_dataset(
                    X_val, y_val, metadata_val,
                    dataset_name="KorJap_2016_2020",
                    station_info=station_info
                )
                logger.info(f"Korea/Japan 2016-2020 validation completed.")
            
            # PANGAEA validation
            if data['val_pangaea_path'].exists():
                logger.info("\nValidating on PANGAEA dataset...")
                pangaea_df = pd.read_csv(data['val_pangaea_path'])
                
                X_pangaea = pangaea_df[data['features']].values
                y_pangaea = pangaea_df['target'].values if 'target' in pangaea_df else pangaea_df['srad'].values
                metadata_pangaea = pangaea_df[['station_id', 'date', 'srad', 'theoretical']].copy()
                
                pangaea_results = validator.validate_dataset(
                    X_pangaea, y_pangaea, metadata_pangaea,
                    dataset_name="PANGAEA",
                    station_info=station_info
                )
                logger.info(f"PANGAEA validation completed.")
        
        # Save final results
        final_results = {
            'experiment_id': experiment_id,
            'combination_name': combination_name,
            'completed': datetime.now().isoformat(),
            'cv_results': cv_results,
            'configuration': config_save
        }
        
        results_path = output_dir / "final_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"Experiment {experiment_id} ({combination_name}) completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 60)
        
        return cv_results
        
    except Exception as e:
        logger.error(f"Error in experiment {experiment_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def run_all_combinations(experiment_id: str,
                         base_dir: Path,
                         output_dir: Optional[Path],
                         use_batch_norm: bool,
                         epochs: int,
                         batch_size: int,
                         patience: int,
                         cv_threshold: float):
    """
    Run all architecture x dropout combinations defined in ModelArchitectures
    Results are saved in subfolders: results/{exp_id}/{combination_name}/
    """
    combinations = ModelArchitectures.get_all_combinations()
    total = len(combinations)
    
    # Set base output directory
    if output_dir is None:
        base_output = base_dir / "results" / experiment_id
    else:
        base_output = Path(output_dir)
    
    print("=" * 60)
    print(f"LUMEN Batch Run: {experiment_id}")
    print(f"Total combinations: {total}")
    print(f"Architectures: {list(ModelArchitectures.ARCHITECTURES.keys())}")
    print(f"Dropout rates: {ModelArchitectures.DROPOUT_RATES}")
    print(f"Output base: {base_output}")
    print("=" * 60)
    
    all_results = {}
    
    for idx, combo in enumerate(combinations, 1):
        combo_name = combo['name']
        print(f"\n{'#' * 60}")
        print(f"# Combination {idx}/{total}: {combo_name}")
        print(f"#   Architecture: {combo['architecture']} -> {combo['hidden_layers']}")
        print(f"#   Dropout: {combo['dropout_rate']}")
        print(f"{'#' * 60}\n")
        
        try:
            cv_results = run_experiment(
                experiment_id=experiment_id,
                base_dir=base_dir,
                output_dir=base_output,
                hidden_layers=combo['hidden_layers'],
                dropout_rate=combo['dropout_rate'],
                combination_name=combo_name,
                use_batch_norm=use_batch_norm,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                cv_threshold=cv_threshold
            )
            
            all_results[combo_name] = {
                'status': 'completed',
                'best_fold': cv_results.get('best_fold', -1),
                'best_mRMSE': cv_results.get('best_validation_mRMSE', float('inf'))
            }
            
        except Exception as e:
            print(f"ERROR in {combo_name}: {e}")
            all_results[combo_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Print final summary
    print("\n" + "=" * 60)
    print("BATCH RUN SUMMARY")
    print("=" * 60)
    
    completed = {k: v for k, v in all_results.items() if v['status'] == 'completed'}
    failed = {k: v for k, v in all_results.items() if v['status'] == 'failed'}
    
    print(f"Completed: {len(completed)}/{total}")
    print(f"Failed: {len(failed)}/{total}")
    
    if completed:
        # Sort by best mRMSE
        ranked = sorted(completed.items(), key=lambda x: x[1]['best_mRMSE'])
        print("\nRanking (by best validation mRMSE):")
        for rank, (name, result) in enumerate(ranked, 1):
            print(f"  {rank}. {name}: mRMSE={result['best_mRMSE']:.4f} (fold {result['best_fold']})")
    
    if failed:
        print("\nFailed combinations:")
        for name, result in failed.items():
            print(f"  - {name}: {result['error']}")
    
    # Save batch summary
    summary_path = base_output / "batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'experiment_id': experiment_id,
            'completed_at': datetime.now().isoformat(),
            'total_combinations': total,
            'completed': len(completed),
            'failed': len(failed),
            'results': all_results
        }, f, indent=2)
    
    print(f"\nBatch summary saved to: {summary_path}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LUMEN Model Training Pipeline")
    parser.add_argument("experiment_id", type=str, help="Experiment ID (e.g., exp2)")
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR,
                        help="Base directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: base_dir/results/experiment_id)")
    parser.add_argument("--architecture", type=str, default=None,
                        help="Single architecture name (if omitted, runs all combinations)")
    parser.add_argument("--dropout-rate", type=float, default=0.2,
                        help="Dropout rate (only used with --architecture)")
    parser.add_argument("--use-batch-norm", action="store_true",
                        help="Use batch normalization")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--cv-threshold", type=float, default=90.0,
                        help="CV_RMSE threshold")
    
    args = parser.parse_args()
    
    if args.architecture:
        # Single architecture mode
        run_experiment(
            experiment_id=args.experiment_id,
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            architecture=args.architecture,
            dropout_rate=args.dropout_rate,
            combination_name=f"{args.architecture}_dropout{int(args.dropout_rate*10)}",
            use_batch_norm=args.use_batch_norm,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            cv_threshold=args.cv_threshold
        )
    else:
        # All combinations mode
        run_all_combinations(
            experiment_id=args.experiment_id,
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            use_batch_norm=args.use_batch_norm,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            cv_threshold=args.cv_threshold
        )
