"""
Configuration validator and type converter for federated learning
"""
import torch
import torch.optim as optim
import yaml
from pathlib import Path
from typing import Dict, Any, Union
import logging

def convert_numeric_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively convert string numeric values to proper types
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with converted numeric values
    """
    if isinstance(config, dict):
        converted = {}
        for key, value in config.items():
            converted[key] = convert_numeric_values(value)
        return converted
    elif isinstance(config, list):
        return [convert_numeric_values(item) for item in config]
    elif isinstance(config, str):
        # Try to convert string numbers to appropriate types
        try:
            # Try int first
            if config.isdigit() or (config.startswith('-') and config[1:].isdigit()):
                return int(config)
            # Try float
            return float(config)
        except ValueError:
            # Return as string if conversion fails
            return config
    else:
        return config

def validate_and_fix_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fix configuration values
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Validated and fixed configuration
    """
    # Convert numeric values
    config = convert_numeric_values(config)
    
    # Ensure required sections exist with defaults
    if 'model' not in config:
        config['model'] = {}
    
    if 'training' not in config:
        config['training'] = {}
    
    if 'federated' not in config:
        config['federated'] = {}
    
    if 'optimizer' not in config:
        config['optimizer'] = {}
    
    if 'privacy' not in config:
        config['privacy'] = {}
    
    if 'data' not in config:
        config['data'] = {}
    
    # Fix specific known issues
    
    # Learning rates - ensure they are floats
    if 'learning_rate' in config['training']:
        config['training']['learning_rate'] = float(config['training']['learning_rate'])
    
    if 'client_lr' in config['optimizer']:
        config['optimizer']['client_lr'] = float(config['optimizer']['client_lr'])
    
    if 'server_lr' in config['optimizer']:
        config['optimizer']['server_lr'] = float(config['optimizer']['server_lr'])
    
    # Weight decay - ensure float
    if 'weight_decay' in config['training']:
        config['training']['weight_decay'] = float(config['training']['weight_decay'])
    
    # Temperature - ensure float
    if 'temperature' in config['training']:
        config['training']['temperature'] = float(config['training']['temperature'])
    
    # Alpha and beta - ensure floats
    if 'alpha' in config['training']:
        config['training']['alpha'] = float(config['training']['alpha'])
    
    if 'beta' in config['training']:
        config['training']['beta'] = float(config['training']['beta'])
    
    # Privacy parameters - ensure floats
    if 'gaussian_noise' in config['privacy'] and 'sigma' in config['privacy']['gaussian_noise']:
        config['privacy']['gaussian_noise']['sigma'] = float(config['privacy']['gaussian_noise']['sigma'])
    
    if 'differential_privacy' in config['privacy']:
        if 'epsilon' in config['privacy']['differential_privacy']:
            config['privacy']['differential_privacy']['epsilon'] = float(config['privacy']['differential_privacy']['epsilon'])
        if 'delta' in config['privacy']['differential_privacy']:
            config['privacy']['differential_privacy']['delta'] = float(config['privacy']['differential_privacy']['delta'])
    
    # Data distribution parameter
    if 'alpha' in config['data']:
        config['data']['alpha'] = float(config['data']['alpha'])
    
    # Loss weights - ensure floats
    if 'loss_weights' in config:
        if 'server' in config['loss_weights']:
            for key in ['lambda1', 'lambda2']:
                if key in config['loss_weights']['server']:
                    config['loss_weights']['server'][key] = float(config['loss_weights']['server'][key])
        
        if 'client' in config['loss_weights']:
            for key in ['mu1', 'mu2']:
                if key in config['loss_weights']['client']:
                    config['loss_weights']['client'][key] = float(config['loss_weights']['client'][key])
    
    # Ensure integer values are integers
    int_fields = [
        ('model', 'embed_dim'),
        ('model', 'num_layers'),
        ('model', 'vocab_size'),
        ('model', 'max_seq_length'),
        ('model', 'split_layer'),
        ('training', 'num_rounds'),
        ('training', 'local_epochs'),
        ('training', 'batch_size'),
        ('training', 'fedavg_frequency'),
        ('federated', 'num_clients'),
        ('privacy', 'quantization', 'bits'),
        ('data', 'max_samples_per_dataset'),
        ('data', 'combined_dataset_size')
    ]
    
    for field_path in int_fields:
        current = config
        for i, key in enumerate(field_path[:-1]):
            if key in current:
                current = current[key]
            else:
                break
        else:
            # All intermediate keys exist
            final_key = field_path[-1]
            if final_key in current:
                try:
                    current[final_key] = int(current[final_key])
                except (ValueError, TypeError):
                    pass  # Keep original value if conversion fails
    
    return config

def load_and_validate_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate and fix
    config = validate_and_fix_config(config)
    
    # Log validation info
    logger = logging.getLogger("ConfigValidator")
    logger.info(f"Configuration loaded and validated from: {config_path}")
    logger.info(f"Learning rate: {config.get('optimizer', {}).get('client_lr', 'Not set')} (type: {type(config.get('optimizer', {}).get('client_lr', 'Not set'))})")
    logger.info(f"Weight decay: {config.get('training', {}).get('weight_decay', 'Not set')} (type: {type(config.get('training', {}).get('weight_decay', 'Not set'))})")
    
    return config

def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration with proper types
    
    Returns:
        Default configuration dictionary
    """
    return {
        'model': {
            'name': "microsoft/DialoGPT-medium",
            'embed_dim': 1024,
            'num_layers': 24,
            'vocab_size': 50257,
            'max_seq_length': 512,
            'split_layer': 6,
            'client': {
                'head_dim': 1024,
                'local_body_layers': 3,
                'tail_layers': 2,
                'dropout': 0.1
            },
            'server': {
                'body_start_layer': 6,
                'body_end_layer': 22,
                'tail_layers': 2
            }
        },
        'training': {
            'num_rounds': 10,
            'local_epochs': 2,
            'batch_size': 4,
            'learning_rate': 5e-5,
            'weight_decay': 0.01,
            'gradient_clip_norm': 1.0,
            'warmup_steps': 100,
            'fedavg_frequency': 5,
            'temperature': 3.0,
            'alpha': 0.7,
            'beta': 0.3
        },
        'loss_weights': {
            'server': {
                'lambda1': 0.5,
                'lambda2': 0.5
            },
            'client': {
                'mu1': 0.5,
                'mu2': 0.5
            }
        },
        'privacy': {
            'gaussian_noise': {
                'sigma': 0.1,
                'enable': True
            },
            'quantization': {
                'bits': 8,
                'enable': True
            },
            'differential_privacy': {
                'epsilon': 1.0,
                'delta': 1e-5
            }
        },
        'federated': {
            'num_clients': 3,
            'client_selection': "all",
            'min_available_clients': 2
        },
        'data': {
            'dataset_source': "real_medical",
            'datasets_to_use': ["pubmed_qa", "medmcqa"],
            'max_samples_per_dataset': 200,
            'combined_dataset_size': 1000,
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1,
            'iid': False,
            'alpha': 0.5,
            'max_seq_length': 512,
            'include_context': True
        },
        'optimizer': {
            'name': "AdamW",
            'server_lr': 1e-4,
            'client_lr': 5e-5,
            'momentum': 0.9,
            'eps': 1e-8
        },
        'evaluation': {
            'eval_frequency': 5,
            'metrics': ["accuracy", "f1", "bleu", "rouge", "perplexity"],
            'save_best_model': True
        },
        'logging': {
            'log_level': "INFO",
            'log_file': "results/logs/training.log",
            'wandb_project': "federated-medical-qa-real-data",
            'save_frequency': 10
        },
        'device': "cuda",
        'mixed_precision': True,
        'random_seed': 42
    }

# Test function
def test_config_validation():
    """Test configuration validation"""
    print("Testing configuration validation...")
    
    # Test with string values that should be converted
    test_config = {
        'training': {
            'learning_rate': '5e-5',
            'weight_decay': '0.01',
            'num_rounds': '10',
            'batch_size': '4'
        },
        'optimizer': {
            'client_lr': '1e-4',
            'server_lr': '5e-5'
        },
        'privacy': {
            'gaussian_noise': {
                'sigma': '0.1'
            }
        }
    }
    
    print("Before validation:")
    print(f"  learning_rate: {test_config['training']['learning_rate']} (type: {type(test_config['training']['learning_rate'])})")
    print(f"  client_lr: {test_config['optimizer']['client_lr']} (type: {type(test_config['optimizer']['client_lr'])})")
    print(f"  num_rounds: {test_config['training']['num_rounds']} (type: {type(test_config['training']['num_rounds'])})")
    
    validated_config = validate_and_fix_config(test_config)
    
    print("After validation:")
    print(f"  learning_rate: {validated_config['training']['learning_rate']} (type: {type(validated_config['training']['learning_rate'])})")
    print(f"  client_lr: {validated_config['optimizer']['client_lr']} (type: {type(validated_config['optimizer']['client_lr'])})")
    print(f"  num_rounds: {validated_config['training']['num_rounds']} (type: {type(validated_config['training']['num_rounds'])})")
    
    print("✓ Configuration validation test completed!")

if __name__ == "__main__":
    test_config_validation()
