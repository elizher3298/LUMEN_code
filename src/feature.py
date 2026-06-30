"""
Feature engineering module for LUMEN model
Creates features based on configuration in feature.yaml
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering class for creating model inputs"""
    
    def __init__(self, config_path: str = "configs/feature.yaml"):
        """
        Initialize feature engineer with configuration
        
        Args:
            config_path: Path to feature configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load feature configuration from YAML"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Feature config not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded feature config from {self.config_path}")
        return config
    
    def calculate_solar_angle(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate solar zenith angle based on date, latitude, longitude
        
        Args:
            df: DataFrame with 'date', 'lat' columns
            
        Returns:
            Series with solar zenith angle in degrees
        """
        # Convert date to day of year
        dates = pd.to_datetime(df['date'])
        day_of_year = dates.dt.dayofyear
        
        # Solar declination angle (simplified)
        declination = 23.45 * np.sin(np.radians((360 * (284 + day_of_year) / 365)))
        
        # Hour angle (assuming solar noon for daily data)
        hour_angle = 0  # Solar noon
        
        # Solar zenith angle
        lat_rad = np.radians(df['lat'])
        dec_rad = np.radians(declination)
        
        cos_zenith = (np.sin(lat_rad) * np.sin(dec_rad) + 
                     np.cos(lat_rad) * np.cos(dec_rad) * np.cos(np.radians(hour_angle)))
        
        # Convert to zenith angle in degrees
        zenith_angle = np.degrees(np.arccos(np.clip(cos_zenith, -1, 1)))
        
        # Return elevation angle (90 - zenith)
        elevation_angle = 90 - zenith_angle
        
        return elevation_angle
    
    def create_derived_feature(self, df: pd.DataFrame, 
                              feature_name: str, 
                              feature_config: Dict) -> pd.Series:
        """
        Create a derived feature based on configuration
        
        Args:
            df: Input DataFrame
            feature_name: Name of the feature to create
            feature_config: Configuration for the feature
            
        Returns:
            Series with the derived feature
        """
        feature_type = feature_config.get('type')
        
        if feature_type == 'ratio':
            numerator = df[feature_config['numerator']]
            denominator = df[feature_config['denominator']]
            # Avoid division by zero
            result = numerator / denominator.replace(0, np.nan)
            return result
            
        elif feature_type == 'difference':
            minuend = df[feature_config['minuend']]
            subtrahend = df[feature_config['subtrahend']]
            return minuend - subtrahend
            
        elif feature_type == 'product':
            factor1 = df[feature_config['factor1']]
            factor2 = df[feature_config['factor2']]
            return factor1 * factor2
            
        elif feature_type == 'transform':
            source = df[feature_config['source']]
            method = feature_config.get('method')
            if method == 'sqrt':
                return np.sqrt(np.maximum(source, 0))
            elif method == 'log1p':
                return np.log1p(np.maximum(source, 0))
            else:
                raise ValueError(f"Unknown transform method: {method}")
            
        elif feature_type == 'calculated':
            method = feature_config.get('method')
            if method == 'solar_position':
                return self.calculate_solar_angle(df)
            else:
                raise ValueError(f"Unknown calculation method: {method}")
                
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def create_features(self, df: pd.DataFrame, 
                       feature_set: str = None,
                       feature_list: List[str] = None) -> pd.DataFrame:
        """
        Create features based on feature set or feature list
        
        Args:
            df: Input DataFrame with raw data
            feature_set: Name of the feature set to use (deprecated)
            feature_list: List of features to create
            
        Returns:
            DataFrame with selected features
        """
        # Handle backward compatibility
        if feature_list is None:
            if feature_set is not None and 'feature_sets' in self.config:
                if feature_set in self.config['feature_sets']:
                    feature_list = self.config['feature_sets'][feature_set]['features']
                else:
                    raise ValueError(f"Unknown feature set: {feature_set}")
            else:
                raise ValueError("Either feature_set or feature_list must be provided")
        
        logger.info(f"Creating features: {feature_list}")
        
        # Start with empty DataFrame
        result_df = pd.DataFrame(index=df.index)
        
        # Add metadata columns (always keep these)
        metadata_cols = ['station_id', 'station_name', 'date', 'lat', 'lon', 'alt', 'dist_to_coast_km', 'srad', 'theoretical']
        for col in metadata_cols:
            if col in df.columns:
                result_df[col] = df[col]
        
        # Process each feature
        for feature_name in feature_list:
            if feature_name in df.columns:
                # Raw feature - just copy
                result_df[feature_name] = df[feature_name]
                logger.debug(f"Added raw feature: {feature_name}")
                
            elif feature_name in self.config.get('derived_features', {}):
                # Derived feature - calculate
                feature_config = self.config['derived_features'][feature_name]
                try:
                    result_df[feature_name] = self.create_derived_feature(
                        df, feature_name, feature_config
                    )
                    logger.debug(f"Created derived feature: {feature_name}")
                except Exception as e:
                    logger.error(f"Failed to create feature {feature_name}: {e}")
                    result_df[feature_name] = np.nan
            else:
                logger.warning(f"Feature {feature_name} not found in raw or derived features")
                result_df[feature_name] = np.nan
        
        return result_df
    
    def create_target(self, df: pd.DataFrame, 
                     target_name: str = 'srad') -> pd.Series:
        """
        Create target variable
        
        Args:
            df: Input DataFrame
            target_name: Name of the target configuration
            
        Returns:
            Series with target values
        """
        if target_name not in self.config['targets']:
            raise ValueError(f"Unknown target: {target_name}")
        
        target_config = self.config['targets'][target_name]
        target_type = target_config.get('type')
        
        if target_type == 'raw':
            return df[target_config['column']]
            
        elif target_type == 'ratio':
            numerator = df[target_config['numerator']]
            denominator = df[target_config['denominator']]
            return numerator / denominator.replace(0, np.nan)
            
        elif target_type == 'difference':
            minuend = df[target_config['minuend']]
            subtrahend = df[target_config['subtrahend']]
            return minuend - subtrahend
            
        else:
            raise ValueError(f"Unknown target type: {target_type}")
    
    def prepare_data(self, df: pd.DataFrame,
                    feature_set: str = None,
                    feature_list: List[str] = None,
                    target_name: str = 'srad',
                    drop_na: bool = True) -> tuple:
        """
        Prepare complete dataset with features and target
        
        Args:
            df: Input DataFrame
            feature_set: Feature set to use (deprecated)
            feature_list: List of features to create
            target_name: Target variable to use
            drop_na: Whether to drop rows with NA values
            
        Returns:
            Tuple of (feature_df, target_series, metadata_df)
        """
        # Create features
        feature_df = self.create_features(df, feature_set=feature_set, feature_list=feature_list)
        
        # Create target
        target = self.create_target(df, target_name)
        
        # Separate metadata and features
        metadata_cols = ['station_id', 'station_name', 'date', 'lat', 'lon', 'alt', 'dist_to_coast_km']
        metadata_df = pd.DataFrame(index=feature_df.index)
        for col in metadata_cols:
            if col in feature_df.columns:
                metadata_df[col] = feature_df[col]
        
        # Get actual feature columns (use provided list or get from feature_set)
        if feature_list is None and feature_set is not None:
            if 'feature_sets' in self.config and feature_set in self.config['feature_sets']:
                feature_list = self.config['feature_sets'][feature_set]['features']
        
        if feature_list is None:
            raise ValueError("No features specified")
            
        X = feature_df[feature_list].copy()
        
        # Add target and reference data to metadata
        metadata_df['target'] = target
        metadata_df['srad_original'] = df['srad']
        metadata_df['theoretical'] = df['theoretical']
        
        # Drop NA if requested
        if drop_na:
            # Find rows with any NA in features or target
            mask = ~(X.isna().any(axis=1) | target.isna())
            X = X[mask]
            target = target[mask]
            metadata_df = metadata_df[mask]
            
            logger.info(f"After dropping NA: {len(X)} samples remain")
        
        return X, target, metadata_df
    
    def get_feature_names(self, feature_set: str) -> List[str]:
        """Get list of feature names for a feature set"""
        if feature_set not in self.config['feature_sets']:
            raise ValueError(f"Unknown feature set: {feature_set}")
        return self.config['feature_sets'][feature_set]['features']
    
    def get_available_feature_sets(self) -> List[str]:
        """Get list of available feature sets"""
        return list(self.config['feature_sets'].keys())
    
    def get_available_targets(self) -> List[str]:
        """Get list of available targets"""
        return list(self.config['targets'].keys())


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load sample data
    data_path = str(Path(__file__).resolve().parent.parent / "data" / "training_korjap.csv")
    df = pd.read_csv(data_path, nrows=1000)  # Load sample
    
    # Initialize feature engineer
    fe = FeatureEngineer("configs/feature.yaml")
    
    # Show available options
    print("Available feature sets:", fe.get_available_feature_sets())
    print("Available targets:", fe.get_available_targets())
    
    # Prepare data with different feature sets
    for feature_set in ['basic', 'normalized', 'advanced']:
        print(f"\n{feature_set.upper()} Feature Set:")
        X, y, metadata = fe.prepare_data(df, feature_set=feature_set, target_name='srad_norm')
        print(f"Features shape: {X.shape}")
        print(f"Features: {list(X.columns)}")
        print(f"Target shape: {y.shape}")
        print(f"Target mean: {y.mean():.3f}")
