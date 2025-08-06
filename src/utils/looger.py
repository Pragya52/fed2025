"""
Logging utilities for federated learning training
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_format: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration for the federated learning system
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
        log_format: Custom log format (optional)
        
    Returns:
        Configured logger instance
    """
    
    # Default log format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Create main logger
    logger = logging.getLogger("FederatedLearning")
    logger.info(f"Logging initialized with level: {log_level}")
    
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter for better console output
    """
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            record.levelname = colored_levelname
        
        # Format the message
        formatted = super().format(record)
        
        # Reset levelname for other handlers
        record.levelname = levelname
        
        return formatted

def setup_colored_logging(log_level: str = "INFO", 
                         log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging with colored console output
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Colored console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    colored_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(colored_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (without colors)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    logger = logging.getLogger("FederatedLearning")
    logger.info(f"Colored logging initialized with level: {log_level}")
    
    return logger

def log_system_info():
    """Log system information"""
    logger = get_logger("SystemInfo")
    
    import platform
    import psutil
    import torch
    
    logger.info("=== SYSTEM INFORMATION ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"CPU: {platform.processor()}")
    logger.info(f"CPU Cores: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # CUDA/GPU information
    if torch.cuda.is_available():
        logger.info(f"CUDA Available: Yes")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.info("CUDA Available: No")
    
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info("=" * 50)

def log_training_config(config: dict):
    """Log training configuration"""
    logger = get_logger("Config")
    
    logger.info("=== TRAINING CONFIGURATION ===")
    
    # Model config
    if 'model' in config:
        logger.info("Model Configuration:")
        for key, value in config['model'].items():
            logger.info(f"  {key}: {value}")
    
    # Training config
    if 'training' in config:
        logger.info("Training Configuration:")
        for key, value in config['training'].items():
            logger.info(f"  {key}: {value}")
    
    # Federated config
    if 'federated' in config:
        logger.info("Federated Configuration:")
        for key, value in config['federated'].items():
            logger.info(f"  {key}: {value}")
    
    # Privacy config
    if 'privacy' in config:
        logger.info("Privacy Configuration:")
        for key, value in config['privacy'].items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for subkey, subvalue in value.items():
                    logger.info(f"    {subkey}: {subvalue}")
            else:
                logger.info(f"  {key}: {value}")
    
    logger.info("=" * 50)

def log_dataset_info(dataset_stats: dict):
    """Log dataset information"""
    logger = get_logger("Dataset")
    
    logger.info("=== DATASET INFORMATION ===")
    for key, value in dataset_stats.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 50)

def log_training_progress(round_num: int, total_rounds: int, metrics: dict):
    """Log training progress"""
    logger = get_logger("Training")
    
    progress = (round_num / total_rounds) * 100
    
    logger.info(f"=== ROUND {round_num}/{total_rounds} ({progress:.1f}%) ===")
    
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            logger.info(f"{metric_name}: {metric_value:.4f}")
        else:
            logger.info(f"{metric_name}: {metric_value}")
    
    logger.info("=" * 40)

def log_client_metrics(client_id: int, metrics: dict):
    """Log client-specific metrics"""
    logger = get_logger(f"Client-{client_id}")
    
    logger.info(f"Client {client_id} Metrics:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        else:
            logger.info(f"  {metric_name}: {metric_value}")

def log_server_metrics(metrics: dict):
    """Log server-specific metrics"""
    logger = get_logger("Server")
    
    logger.info("Server Metrics:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        else:
            logger.info(f"  {metric_name}: {metric_value}")

def log_privacy_analysis(privacy_metrics: dict):
    """Log privacy analysis"""
    logger = get_logger("Privacy")
    
    logger.info("=== PRIVACY ANALYSIS ===")
    for key, value in privacy_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"{key}: {value:.6f}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("=" * 50)

def log_federated_summary(summary: dict):
    """Log federated learning summary"""
    logger = get_logger("FedSummary")
    
    logger.info("=== FEDERATED LEARNING SUMMARY ===")
    
    if 'server_info' in summary:
        logger.info("Server Information:")
        for key, value in summary['server_info'].items():
            logger.info(f"  {key}: {value}")
    
    if 'privacy_analysis' in summary:
        logger.info("Privacy Analysis:")
        for key, value in summary['privacy_analysis'].items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.6f}")
            else:
                logger.info(f"  {key}: {value}")
    
    if 'communication_stats' in summary:
        logger.info("Communication Statistics:")
        for key, value in summary['communication_stats'].items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")
    
    logger.info("=" * 50)

# Convenience function for quick setup
def quick_setup(log_level: str = "INFO", use_colors: bool = True, log_file: Optional[str] = None) -> logging.Logger:
    """
    Quick logging setup with sensible defaults
    
    Args:
        log_level: Logging level
        use_colors: Whether to use colored output
        log_file: Optional log file
        
    Returns:
        Configured logger
    """
    if use_colors:
        return setup_colored_logging(log_level, log_file)
    else:
        return setup_logging(log_level, log_file)

# Test function
def test_logging():
    """Test the logging setup"""
    
    # Setup colored logging
    logger = quick_setup("DEBUG", use_colors=True, log_file="test.log")
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test specialized logging functions
    log_system_info()
    
    config = {
        'model': {'embed_dim': 1024, 'num_layers': 24},
        'training': {'num_rounds': 100, 'batch_size': 8},
        'privacy': {'gaussian_noise': {'sigma': 0.1}}
    }
    log_training_config(config)
    
    metrics = {'accuracy': 0.85, 'loss': 0.32, 'f1_score': 0.78}
    log_training_progress(5, 100, metrics)
    
    print("Logging test completed!")

if __name__ == "__main__":
    test_logging()
