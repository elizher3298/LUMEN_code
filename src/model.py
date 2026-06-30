#!/usr/bin/env python3
"""
model.py - LUMEN TensorFlow Model Architecture Module
Defines neural network architectures for solar radiation prediction
"""

import tensorflow as tf
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

@dataclass
class ModelConfig:
    """Model configuration"""
    input_size: int
    hidden_layers: List[int]
    dropout_rate: float = 0.2
    use_batch_norm: bool = False
    activation: str = 'relu'
    output_activation: str = 'linear'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation,
            'output_activation': self.output_activation
        }

class ModelArchitectures:
    """Predefined model architectures"""
    
    # Dropout rate options
    #DROPOUT_RATES = [0.2,0.3]
    #DROPOUT_RATES = [0.1,0.3]
    DROPOUT_RATES = [0.1]
    
    # Based on existing config.py architectures
    ARCHITECTURES = {
        #'double_16_16': [16, 16],
        #'double_32_32': [32, 32],
        #'double_64_64': [64, 64]
        #'double_128_128': [128, 128]
        #'double_256_256': [256, 256],
        'triple_46_46_46': [46, 46, 46]
        #'triple_256_128_64': [256, 128, 64],
        #'triple_256_256_256': [256, 256, 256],
        #'quad_64_64_64_64': [64, 64, 64, 64],
        #'quad_256_192_128_64': [256, 192, 128, 64],
        #'quad_256_256_256_256': [256, 256, 256, 256],
        #'five_64_64_64_64_64': [64, 64, 64, 64, 64],
        #'five_256_192_128_96_64': [256, 192, 128, 96, 64],
        #'five_256_256_256_256_256': [256, 256, 256, 256, 256],
        # 'six_64_64_64_64_64_64': [64, 64, 64, 64, 64, 64],
        # 'six_256_213_170_128_96_64': [256, 213, 170, 128, 96, 64],
        # 'six_256_256_256_256_256_256': [256, 256, 256, 256, 256, 256],
        # 'seven_64_64_64_64_64_64_64': [64, 64, 64, 640, 640, 64, 64],
        # 'seven_256_224_192_160_128_96_64': [256, 224, 192, 160, 128, 96, 64],
        # 'seven_256_256_256_256_256_256_256': [256, 256, 256, 256, 256, 256, 256],
        # 'eight_64_64_64_64_64_64_64_64': [64, 64, 64, 64, 64, 64, 64, 64],      
        # 'eight_256_224_192_160_128_112_88_64': [256, 224, 192, 160, 128, 112, 88, 64],
        # 'eight_256_256_256_256_256_256_256_256': [256, 256, 256, 256, 256, 256, 256, 256],
        # 'nine_64_64_64_64_64_64_64_64_64': [64, 64, 64, 64, 64, 64, 64, 64, 64],  
        # 'nine_256_224_192_160_128_112_96_80_64': [256, 224, 192, 160, 128, 112, 96, 80, 64],
        # 'nine_256_256_256_256_256_256_256_256_256': [256, 256, 256, 256, 256, 256, 256, 256, 256]
    }
    
    @classmethod
    def get_architecture(cls, name: str) -> List[int]:
        """Get architecture by name"""
        if name not in cls.ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {name}. Available: {list(cls.ARCHITECTURES.keys())}")
        return cls.ARCHITECTURES[name]

    @classmethod
    def get_all_combinations(cls):
        """arcitecture x dropout combination"""
        combinations = []
        for arch_name, layers in cls.ARCHITECTURES.items():
            for dropout in cls.DROPOUT_RATES:
                combinations.append({
                    'architecture': arch_name,
                    'hidden_layers': layers,
                    'dropout_rate': dropout,
                    'name': f"{arch_name}_dropout{int(dropout*10)}"
                })
        return combinations    

    @classmethod
    def get_default_architecture(cls) -> List[int]:
        """Get default architecture"""
        return cls.ARCHITECTURES['triple_256_192_96']
    
    @classmethod
    def get_dropout_rates(cls) -> List[float]:
        """Get available dropout rates"""
        return cls.DROPOUT_RATES
    
    @classmethod
    def get_default_dropout_rate(cls) -> float:
        """Get default dropout rate"""
        return 0.2
    
    @classmethod
    def get_all_model_variants(cls, architecture_name: str = None) -> List[Dict]:
        """
        Get all model variants with different dropout rates
        
        Args:
            architecture_name: Specific architecture name, or None for all
            
        Returns:
            List of model configuration dictionaries
        """
        variants = []
        
        if architecture_name:
            architectures = {architecture_name: cls.ARCHITECTURES[architecture_name]}
        else:
            architectures = cls.ARCHITECTURES
        
        for arch_name, layers in architectures.items():
            for dropout_rate in cls.DROPOUT_RATES:
                variants.append({
                    'name': f"{arch_name}_dropout{int(dropout_rate*10)}",
                    'architecture': arch_name,
                    'hidden_layers': layers,
                    'dropout_rate': dropout_rate
                })
        
        return variants

def create_model(config: ModelConfig, name: Optional[str] = None) -> tf.keras.Model:
    """
    Create TensorFlow/Keras model for solar radiation prediction
    
    Args:
        config: Model configuration
        name: Optional model name
        
    Returns:
        Compiled TensorFlow model
    """
    model = tf.keras.Sequential(name=name)
    
    # Input layer
    model.add(tf.keras.layers.Input(shape=(config.input_size,)))
    
    # Hidden layers
    for i, units in enumerate(config.hidden_layers):
        # Dense layer
        model.add(tf.keras.layers.Dense(
            units, 
            activation=config.activation,
            kernel_initializer='he_normal',
            name=f'dense_{i+1}'
        ))
        
        # Batch normalization (before dropout)
        if config.use_batch_norm:
            model.add(tf.keras.layers.BatchNormalization(name=f'batch_norm_{i+1}'))
        
        # Dropout
        if config.dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(config.dropout_rate, name=f'dropout_{i+1}'))
    
    # Output layer
    model.add(tf.keras.layers.Dense(
        1, 
        activation=config.output_activation,
        name='output'
    ))
    
    return model

def compile_model(model: tf.keras.Model, 
                  optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
                  loss: str = 'mse',
                  metrics: Optional[List[str]] = None) -> tf.keras.Model:
    """
    Compile TensorFlow model
    
    Args:
        model: TensorFlow model
        optimizer: Optimizer (default: Adam with lr=0.001)
        loss: Loss function (default: 'mse')
        metrics: List of metrics (default: ['mae'])
        
    Returns:
        Compiled model
    """
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    if metrics is None:
        metrics = ['mae']
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model

def create_compiled_model(config: ModelConfig,
                          optimizer_config: Optional[Dict[str, Any]] = None,
                          loss: str = 'mape',
                          metrics: Optional[List[str]] = None) -> tf.keras.Model:
    """
    Create and compile model in one step
    
    Args:
        config: Model configuration
        optimizer_config: Optimizer configuration dict
        loss: Loss function
        metrics: List of metrics
        
    Returns:
        Compiled TensorFlow model
    """
    # Create model
    model = create_model(config)
    
    # Create optimizer
    if optimizer_config is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    else:
        opt_type = optimizer_config.get('type', 'adam').lower()
        learning_rate = optimizer_config.get('learning_rate', 0.001)
        
        if opt_type == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif opt_type == 'sgd':
            momentum = optimizer_config.get('momentum', 0.0)
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        elif opt_type == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif opt_type == 'adamw':
            weight_decay = optimizer_config.get('weight_decay', 0.01)
            optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")
    
    # Compile model
    model = compile_model(model, optimizer, loss, metrics)
    
    return model

def get_model_summary(model: tf.keras.Model) -> Dict[str, Any]:
    """
    Get detailed model summary
    
    Args:
        model: TensorFlow model
        
    Returns:
        Dictionary with model information
    """
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    layer_info = []
    for layer in model.layers:
        layer_dict = {
            'name': layer.name,
            'type': type(layer).__name__,
            'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A',
            'parameters': layer.count_params() if hasattr(layer, 'count_params') else 0
        }
        layer_info.append(layer_dict)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'layer_count': len(model.layers),
        'layers': layer_info
    }

def save_model(model: tf.keras.Model, filepath: str):
    """Save model to file"""
    model.save(filepath)
    logging.info(f"Model saved to {filepath}")

def load_model(filepath: str) -> tf.keras.Model:
    """Load model from file"""
    model = tf.keras.models.load_model(filepath)
    logging.info(f"Model loaded from {filepath}")
    return model

# Example usage
if __name__ == "__main__":
    # Example 1: Create models with different dropout rates
    print("Testing different dropout rates for triple_256_192_96:\n")
    
    for dropout in ModelArchitectures.get_dropout_rates():
        config = ModelConfig(
            input_size=10,  # Number of features
            hidden_layers=ModelArchitectures.get_architecture('triple_256_192_96'),
            dropout_rate=dropout,
            use_batch_norm=False
        )
        
        model = create_compiled_model(config)
        summary = get_model_summary(model)
        print(f"Dropout {dropout}: Total parameters = {summary['total_parameters']:,}")
    
    # Example 2: Get all variants for an architecture
    print("\nAll variants for double_192_96:")
    variants = ModelArchitectures.get_all_model_variants('double_192_96')
    for variant in variants:
        print(f"  {variant['name']}: dropout={variant['dropout_rate']}")
