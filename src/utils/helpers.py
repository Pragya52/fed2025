"""
Utility helper functions for federated learning
"""

import torch
import numpy as np
import random
import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_checkpoint(state: Dict[str, Any], 
                   filepath: Union[str, Path],
                   is_best: bool = False):
    """
    Save model checkpoint
    
    Args:
        state: State dictionary to save
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(state, filepath)
    
    if is_best:
        best_path = filepath.parent / f"best_{filepath.name}"
        torch.save(state, best_path)

def load_checkpoint(filepath: Union[str, Path], 
                   device: str = 'cpu') -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint file
        device: Device to load checkpoint on
        
    Returns:
        Loaded state dictionary
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config: Dict[str, Any], 
               filepath: Union[str, Path]):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        filepath: Path to save configuration
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }

def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Get model size in megabytes
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    return model_size_mb

def create_experiment_dir(base_dir: Union[str, Path], 
                         experiment_name: Optional[str] = None) -> Path:
    """
    Create experiment directory with timestamp
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional experiment name
        
    Returns:
        Path to created experiment directory
    """
    base_dir = Path(base_dir)
    
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    exp_dir = base_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'results').mkdir(exist_ok=True)
    (exp_dir / 'configs').mkdir(exist_ok=True)
    
    return exp_dir

def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"

def compute_metrics_delta(current_metrics: Dict[str, float], 
                         previous_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Compute change in metrics between two measurements
    
    Args:
        current_metrics: Current metrics
        previous_metrics: Previous metrics
        
    Returns:
        Dictionary with metric deltas
    """
    deltas = {}
    
    for key in current_metrics:
        if key in previous_metrics:
            delta = current_metrics[key] - previous_metrics[key]
            deltas[f"{key}_delta"] = delta
            
            # Compute percentage change
            if previous_metrics[key] != 0:
                pct_change = (delta / abs(previous_metrics[key])) * 100
                deltas[f"{key}_pct_change"] = pct_change
    
    return deltas

def aggregate_client_metrics(client_metrics: List[Dict[str, float]], 
                           weights: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Aggregate metrics from multiple clients
    
    Args:
        client_metrics: List of metric dictionaries from clients
        weights: Optional weights for weighted averaging
        
    Returns:
        Aggregated metrics
    """
    if not client_metrics:
        return {}
    
    if weights is None:
        weights = [1.0] * len(client_metrics)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Get all metric names
    all_metric_names = set()
    for metrics in client_metrics:
        all_metric_names.update(metrics.keys())
    
    # Aggregate each metric
    aggregated = {}
    
    for metric_name in all_metric_names:
        values = []
        metric_weights = []
        
        for i, metrics in enumerate(client_metrics):
            if metric_name in metrics:
                values.append(metrics[metric_name])
                metric_weights.append(weights[i])
        
        if values:
            # Weighted average
            weighted_avg = sum(v * w for v, w in zip(values, metric_weights)) / sum(metric_weights)
            aggregated[f"avg_{metric_name}"] = weighted_avg
            
            # Additional statistics
            aggregated[f"min_{metric_name}"] = min(values)
            aggregated[f"max_{metric_name}"] = max(values)
            aggregated[f"std_{metric_name}"] = np.std(values)
    
    return aggregated

def log_system_info(logger: logging.Logger):
    """
    Log system information
    
    Args:
        logger: Logger instance
    """
    import psutil
    import platform
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  CPU: {platform.processor()}")
    logger.info(f"  CPU Cores: {psutil.cpu_count()}")
    logger.info(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name()}")
        logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        logger.info(f"  CUDA Version: {torch.version.cuda}")
    else:
        logger.info("  GPU: Not available")

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if config is valid, False otherwise
    """
    required_sections = [
        'model', 'training', 'federated', 'data', 'privacy'
    ]
    
    for section in required_sections:
        if section not in config:
            print(f"Missing required config section: {section}")
            return False
    
    # Validate model config
    model_required = ['embed_dim', 'vocab_size', 'max_seq_length']
    for key in model_required:
        if key not in config['model']:
            print(f"Missing required model config: {key}")
            return False
    
    # Validate training config
    training_required = ['num_rounds', 'batch_size', 'learning_rate']
    for key in training_required:
        if key not in config['training']:
            print(f"Missing required training config: {key}")
            return False
    
    # Validate federated config
    if 'num_clients' not in config['federated']:
        print("Missing required federated config: num_clients")
        return False
    
    # Validate privacy config
    privacy_required = ['gaussian_noise', 'quantization', 'differential_privacy']
    for key in privacy_required:
        if key not in config['privacy']:
            print(f"Missing required privacy config: {key}")
            return False
    
    return True

def get_device_info() -> Dict[str, Any]:
    """
    Get detailed device information
    
    Returns:
        Dictionary with device information
    """
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cpu_count': torch.get_num_threads()
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(),
            'device_capability': torch.cuda.get_device_capability(),
            'memory_allocated': torch.cuda.memory_allocated(),
            'memory_reserved': torch.cuda.memory_reserved()
        })
    
    return info

def tensor_to_list(tensor: torch.Tensor) -> List:
    """
    Convert tensor to list for serialization
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        List representation of tensor
    """
    return tensor.detach().cpu().tolist()

def list_to_tensor(data: List, 
                  device: str = 'cpu', 
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert list to tensor
    
    Args:
        data: List data
        device: Target device
        dtype: Target data type
        
    Returns:
        PyTorch tensor
    """
    return torch.tensor(data, device=device, dtype=dtype)

def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists
    
    Args:
        directory: Directory path
        
    Returns:
        Path object
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def get_git_hash() -> Optional[str]:
    """
    Get current git commit hash
    
    Returns:
        Git hash string or None if not in git repo
    """
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return None

def create_run_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create run information dictionary
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Run information dictionary
    """
    run_info = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'system_info': get_device_info(),
        'git_hash': get_git_hash(),
        'random_seed': config.get('random_seed', 42)
    }
    
    return run_info

def print_training_progress(round_num: int, 
                          total_rounds: int,
                          metrics: Dict[str, float],
                          elapsed_time: float):
    """
    Print training progress in a formatted way
    
    Args:
        round_num: Current round number
        total_rounds: Total number of rounds
        metrics: Current metrics
        elapsed_time: Elapsed time in seconds
    """
    progress_pct = (round_num / total_rounds) * 100
    time_str = format_time(elapsed_time)
    
    print(f"\n{'='*60}")
    print(f"Round {round_num}/{total_rounds} ({progress_pct:.1f}%) - {time_str}")
    print(f"{'='*60}")
    
    if metrics:
        print("Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    print(f"{'='*60}")

class EarlyStopping:
    """
    Early stopping utility class
    """
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'max'):
        """
        Args:
            patience: Number of rounds to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for maximizing metric, 'min' for minimizing
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current metric score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.min_delta:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop

class MovingAverage:
    """
    Moving average utility class
    """
    
    def __init__(self, window_size: int = 10):
        """
        Args:
            window_size: Size of the moving window
        """
        self.window_size = window_size
        self.values = []
        
    def update(self, value: float) -> float:
        """
        Update with new value and return moving average
        
        Args:
            value: New value
            
        Returns:
            Current moving average
        """
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        
        return sum(self.values) / len(self.values)
    
    def get_average(self) -> float:
        """Get current moving average"""
        return sum(self.values) / len(self.values) if self.values else 0.0
    
    def reset(self):
        """Reset the moving average"""
        self.values = []

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten nested dictionary
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def check_memory_usage() -> Dict[str, float]:
    """
    Check current memory usage
    
    Returns:
        Dictionary with memory usage information
    """
    import psutil
    
    # System memory
    memory = psutil.virtual_memory()
    memory_info = {
        'system_memory_total_gb': memory.total / (1024**3),
        'system_memory_used_gb': memory.used / (1024**3),
        'system_memory_available_gb': memory.available / (1024**3),
        'system_memory_percent': memory.percent
    }
    
    # GPU memory if available
    if torch.cuda.is_available():
        memory_info.update({
            'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
            'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
            'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
        })
    
    return memory_info

def cleanup_memory():
    """Clean up memory"""
    import gc
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def save_metrics_to_csv(metrics_history: List[Dict[str, Any]], 
                       filepath: Union[str, Path]):
    """
    Save metrics history to CSV file
    
    Args:
        metrics_history: List of metrics dictionaries
        filepath: Path to save CSV file
    """
    import pandas as pd
    
    if not metrics_history:
        return
    
    # Flatten nested dictionaries
    flattened_metrics = []
    for metrics in metrics_history:
        flat_metrics = flatten_dict(metrics)
        flattened_metrics.append(flat_metrics)
    
    # Create DataFrame and save
    df = pd.DataFrame(flattened_metrics)
    df.to_csv(filepath, index=False)

def load_metrics_from_csv(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load metrics history from CSV file
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        List of metrics dictionaries
    """
    import pandas as pd
    
    df = pd.read_csv(filepath)
    return df.to_dict('records')

class ProgressTracker:
    """
    Progress tracking utility
    """
    
    def __init__(self, total_steps: int):
        """
        Args:
            total_steps: Total number of steps
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
        self.step_times = []
        
    def update(self, step: int = None):
        """
        Update progress
        
        Args:
            step: Current step (if None, increment by 1)
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        current_time = datetime.now()
        self.step_times.append(current_time)
        
    def get_progress_info(self) -> Dict[str, Any]:
        """
        Get progress information
        
        Returns:
            Progress information dictionary
        """
        if not self.step_times:
            return {}
            
        current_time = datetime.now()
        elapsed_time = (current_time - self.start_time).total_seconds()
        
        progress_pct = (self.current_step / self.total_steps) * 100
        
        # Estimate remaining time
        if self.current_step > 0:
            avg_time_per_step = elapsed_time / self.current_step
            remaining_steps = self.total_steps - self.current_step
            estimated_remaining_time = avg_time_per_step * remaining_steps
        else:
            estimated_remaining_time = 0
        
        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress_percent': progress_pct,
            'elapsed_time_seconds': elapsed_time,
            'estimated_remaining_seconds': estimated_remaining_time,
            'avg_time_per_step': elapsed_time / max(1, self.current_step)
        }

# Test functions
def test_helper_functions():
    """Test helper functions"""
    print("Testing helper functions...")
    
    # Test random seed
    set_random_seed(42)
    random_vals1 = [random.random() for _ in range(5)]
    
    set_random_seed(42)
    random_vals2 = [random.random() for _ in range(5)]
    
    assert random_vals1 == random_vals2, "Random seed not working properly"
    print("✓ Random seed test passed")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=3, mode='max')
    
    scores = [0.1, 0.2, 0.3, 0.25, 0.26, 0.27, 0.28]  # Should not stop
    for score in scores:
        if early_stopping(score):
            break
    assert not early_stopping.early_stop, "Early stopping triggered incorrectly"
    
    scores = [0.5, 0.4, 0.3, 0.2, 0.1]  # Should stop after patience
    early_stopping = EarlyStopping(patience=3, mode='max')
    stop_triggered = False
    for score in scores:
        if early_stopping(score):
            stop_triggered = True
            break
    assert stop_triggered, "Early stopping should have been triggered"
    print("✓ Early stopping test passed")
    
    # Test moving average
    ma = MovingAverage(window_size=3)
    values = [1, 2, 3, 4, 5]
    averages = [ma.update(v) for v in values]
    
    expected_averages = [1.0, 1.5, 2.0, 3.0, 4.0]
    for expected, actual in zip(expected_averages, averages):
        assert abs(expected - actual) < 1e-6, f"Moving average incorrect: {expected} vs {actual}"
    print("✓ Moving average test passed")
    
    # Test flatten dict
    nested_dict = {
        'a': 1,
        'b': {
            'c': 2,
            'd': {
                'e': 3
            }
        }
    }
    
    flattened = flatten_dict(nested_dict)
    expected = {'a': 1, 'b_c': 2, 'b_d_e': 3}
    assert flattened == expected, f"Dict flattening incorrect: {flattened} vs {expected}"
    print("✓ Flatten dict test passed")
    
    # Test progress tracker
    tracker = ProgressTracker(total_steps=10)
    tracker.update(5)
    
    progress_info = tracker.get_progress_info()
    assert progress_info['current_step'] == 5
    assert progress_info['progress_percent'] == 50.0
    print("✓ Progress tracker test passed")
    
    print("All helper function tests passed!")

if __name__ == "__main__":
    test_helper_functions()
