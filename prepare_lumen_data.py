#!/usr/bin/env python3
"""
prepare_lumen_data.py - Prepare LUMEN experiment data based on GPU experiment plan
This script only prepares data according to the experiment configuration, without running the model.
Usage: python prepare_lumen_data.py exp3 --min-station-id 4001
"""
import sys
import pandas as pd
from pathlib import Path
import argparse
import json

sys.path.append(str(Path(__file__).resolve().parent))
from src.feature import FeatureEngineer

def prepare_data(exp_id, min_station_id=None, max_station_id=None):
    """
    Prepare experiment data with optional station filtering
    
    This function reads experiment configuration from GPU experiment plan CSV,
    applies feature engineering, and saves processed data for training/validation.
    
    Args:
        exp_id: Experiment ID (e.g., 'exp3')
        min_station_id: Minimum station ID filter (inclusive)
        max_station_id: Maximum station ID filter (inclusive)
    
    Returns:
        Path: Directory containing prepared experiment data
    """
    BASE = Path(__file__).resolve().parent
    
    # Read experiment configuration from CSV
    plan = pd.read_csv(BASE / "configs/gpu_experiment_plan.csv")
    exp_config = plan[plan['id'] == exp_id].iloc[0]
    
    # Extract features and target from configuration
    features = exp_config['features'].split()
    target = exp_config['target']
    
    print(f"\n=== Experiment Configuration ===")
    print(f"Experiment ID: {exp_id}")
    print(f"Features: {features}")
    print(f"Target: {target}")
    
    # Display station filtering information if applicable
    if min_station_id or max_station_id:
        print(f"\n=== Station Filtering ===")
        if min_station_id:
            print(f"  Minimum Station ID: {min_station_id}")
        if max_station_id:
            print(f"  Maximum Station ID: {max_station_id}")
    
    # Create experiment directory with station filter suffix if needed
    exp_dir = BASE / "experiments" / exp_id
    if min_station_id:
        exp_dir = BASE / "experiments" / f"{exp_id}_station{min_station_id}+"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create symbolic link for compatibility with existing scripts
    if min_station_id:
        link_path = BASE / "experiments" / exp_id
        if not link_path.exists():
            import os
            os.symlink(exp_dir.name, str(link_path))
            print(f"📎 Created symbolic link: {exp_id} → {exp_dir.name}")
    
    # Initialize FeatureEngineer for data processing
    fe = FeatureEngineer(config_path=str(BASE / "configs/feature.yaml"))
    
    # Process each dataset (training, validation Korea/Japan, validation PANGAEA)
    datasets = [
        ("train_data", "training_korjap.csv"),
        ("val_korjap_data", "validation_korjap.csv"),
        ("val_pangaea_data", "validation_pangaea.csv")
    ]
    
    for output_name, source_file in datasets:
        print(f"\n=== Processing {source_file} ===")
        
        # Load raw data
        df = pd.read_csv(BASE / "data" / source_file)
        original_count = len(df)
        unique_stations_before = df['station_id'].nunique()
        
        # Apply station filtering if specified
        if min_station_id is not None:
            df = df[df['station_id'] >= min_station_id]
        if max_station_id is not None:
            df = df[df['station_id'] <= max_station_id]
        
        filtered_count = len(df)
        unique_stations_after = df['station_id'].nunique()
        
        # Skip if no data remains after filtering
        if filtered_count == 0:
            print(f"  ⚠️  Warning: No data remaining after station filtering!")
            continue
        
        # Apply feature engineering using FeatureEngineer
        # This creates the specified features and target variable
        X, y, metadata = fe.prepare_data(
            df,
            feature_list=features,
            target_name=target
        )
        
        # Combine features, metadata, and target into single dataframe
        output_df = pd.concat([metadata, X], axis=1)
        output_df['target'] = y
        
        # Ensure 'srad' column exists for compatibility with main.py
        if 'srad' not in output_df.columns and 'srad_original' in metadata.columns:
            output_df['srad'] = metadata['srad_original']
        
        # Save processed data to experiment directory
        output_file = exp_dir / f"{output_name}.csv"
        output_df.to_csv(output_file, index=False)
        
        # Display processing statistics
        print(f"  Original: {original_count} rows, {unique_stations_before} stations")
        print(f"  Filtered: {filtered_count} rows, {unique_stations_after} stations")
        print(f"  Final: {len(output_df)} rows (after NaN removal)")
        print(f"  Saved to: {output_file}")
        
        # Display station ID range in the output
        if len(output_df) > 0:
            station_ids = output_df['station_id'].unique()
            print(f"  Station ID range: {station_ids.min():.0f} ~ {station_ids.max():.0f}")
    
    # Save filtering metadata for reproducibility
    filter_info = {
        'experiment_id': exp_id,
        'min_station_id': min_station_id,
        'max_station_id': max_station_id,
        'filtered_experiment_id': f"{exp_id}_station{min_station_id}+" if min_station_id else exp_id,
        'features': features,
        'target': target
    }
    
    with open(exp_dir / 'filter_info.json', 'w') as f:
        json.dump(filter_info, f, indent=2)
    
    print(f"\n=== Data Preparation Complete ===")
    print(f"Output directory: {exp_dir}")
    
    return exp_dir

def main():
    """
    Main function to handle command line arguments and execute data preparation
    """
    parser = argparse.ArgumentParser(
        description='Prepare LUMEN experiment data based on GPU experiment plan',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Prepare data for experiment 3 with default settings
    python prepare_lumen_data.py exp3
    
    # Prepare data excluding stations with ID below 4001
    python prepare_lumen_data.py exp3 --min-station-id 4001
    
    # Prepare data for specific station ID range
    python prepare_lumen_data.py exp3 --min-station-id 1000 --max-station-id 5000
        """
    )
    
    # Required argument: experiment ID
    parser.add_argument('exp_id', 
                       help='Experiment ID from gpu_experiment_plan.csv (e.g., exp2, exp3)')
    
    # Optional station filtering arguments
    parser.add_argument('--min-station-id', 
                       type=int, 
                       default=999,
                       help='Minimum station ID to include (default: no minimum)')
    
    parser.add_argument('--max-station-id', 
                       type=int, 
                       default=None,
                       help='Maximum station ID to include (default: no maximum)')
    
    args = parser.parse_args()
    
    # Execute data preparation
    try:
        exp_dir = prepare_data(args.exp_id, args.min_station_id, args.max_station_id)
        print(f"\n✅ Success! Data prepared in: {exp_dir}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()