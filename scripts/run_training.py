#!/usr/bin/env python3
"""
Main Training Script for Federated Split Learning with Medical QA
Uses Real LLaMA-2-7B Model from Hugging Face
"""

import os
import sys
import argparse
import yaml
import logging
import torch
import numpy as np
import wandb
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Hugging Face Transformers imports for real LLaMA-2
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Import federated learning components (we'll adapt these for real LLaMA-2)
from src.federated.client import ClientManager
from src.federated.server import FederatedServer, ServerCommunicationHandler
from src.data.medical_qa_loader import MedicalQADataLoader, create_federated_dataloaders
from src.training.metrics import MedicalQAMetrics, FederatedMetricsAggregator

try:
    from src.utils.logger import setup_logging
except ImportError:
    print("Warning: Could not import logger, using basic logging")
    def setup_logging(log_level="INFO", log_file=None):
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )
        return logging.getLogger("FederatedLearning")

try:
    from src.utils.helpers import set_random_seed, save_checkpoint, load_checkpoint
except ImportError:
    print("Warning: Could not import helpers")
    def set_random_seed(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def save_checkpoint(state, filepath):
        torch.save(state, filepath)
    
    def load_checkpoint(filepath):
        return torch.load(filepath)

def create_default_config():
    return {
        'model': {
            'model_name': 'meta-llama/Llama-2-7b-hf',  # Real LLaMA-2-7B
            'use_real_llama': True,  # Flag to use real LLaMA
            'max_seq_length': 512, 
            'load_in_8bit': False,  # Set to True to reduce memory usage
            'load_in_4bit': False,  # Set to True for even more memory savings
            'use_flash_attention': False,  # Enable for better performance
        },
        'training': {
            'num_rounds': 10, 
            'local_epochs': 1,  # Reduced for LLaMA-2 (it's large)
            'batch_size': 1,  # Small batch size for 7B model
            'gradient_accumulation_steps': 8,  # Accumulate gradients
            'learning_rate': 5e-5, 
            'weight_decay': 0.01,
            'warmup_steps': 100,
            'max_grad_norm': 1.0,
            'save_steps': 500,
            'eval_steps': 500,
            'logging_steps': 10,
        },
        'federated': {'num_clients': 3},
        'optimizer': {
            'name': 'AdamW', 
            'client_lr': 2e-5,  # Lower LR for large model
            'server_lr': 1e-5,  # Even lower for server
            'eps': 1e-8,
            'betas': [0.9, 0.999],
        },
        'privacy': {
            'target_epsilon': 10.0,
            'gaussian_noise': {'sigma': 0.1}, 
            'quantization': {'bits': 8},
            'enable_privacy': True
        },
        'data': {
            'dataset_name': 'combined',
            'combined_dataset_size': 1000,
            'train_split': 0.8, 
            'val_split': 0.1, 
            'test_split': 0.1, 
            'iid': False, 
            'alpha': 0.5
        },
        'loss': {
            'loss_type': 'causal_lm',  # Standard causal language modeling
            'temperature': 3.0,
            'distillation_weight': 0.5,
            'task_weight': 0.5,
            'medical_weight': 0.1,
        },
        'logging': {'log_level': 'INFO', 'log_file': 'training.log'},
        'evaluation': {'eval_frequency': 1},
        'debug_mode': False
    }

def load_and_validate_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure we're using real LLaMA
    config.setdefault('model', {})['use_real_llama'] = True
    config.setdefault('model', {})['model_name'] = 'meta-llama/Llama-2-7b-hf'
    
    # Basic type conversion for critical values
    if 'optimizer' in config:
        for lr_key in ['client_lr', 'server_lr']:
            if lr_key in config['optimizer']:
                config['optimizer'][lr_key] = float(config['optimizer'][lr_key])
    
    if 'training' in config:
        for float_key in ['learning_rate', 'weight_decay']:
            if float_key in config['training']:
                config['training'][float_key] = float(config['training'][float_key])
    
    return config

def load_llama2_model_and_tokenizer(model_name: str, device: str, config: Dict):
    """Load real LLaMA-2 model and tokenizer from Hugging Face"""
    
    print(f"🦙 Loading LLaMA-2 model: {model_name}")
    print("This may take several minutes for the 7B model...")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"  # Important for generation
        )
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✓ Tokenizer loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        print("💡 You may need to request access to LLaMA-2 on Hugging Face")
        raise
    
    # Load model with appropriate settings
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,  # Use half precision
    }
    
    # Memory optimization options
    if config.get('model', {}).get('load_in_8bit', False):
        model_kwargs["load_in_8bit"] = True
        print("📉 Using 8-bit quantization to reduce memory usage")
        
    if config.get('model', {}).get('load_in_4bit', False):
        model_kwargs["load_in_4bit"] = True
        print("📉 Using 4-bit quantization for maximum memory savings")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move to device if not using quantization
        if not (config.get('model', {}).get('load_in_8bit', False) or 
                config.get('model', {}).get('load_in_4bit', False)):
            model = model.to(device)
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ LLaMA-2 model loaded successfully")
        print(f"📊 Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,}")
        print(f"💾 Model size: ~{total_params * 2 / 1e9:.1f} GB (fp16)")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Possible solutions:")
        print("   1. Request access to LLaMA-2 on Hugging Face Hub")
        print("   2. Use load_in_8bit or load_in_4bit for memory issues")
        print("   3. Ensure you have enough GPU memory (7B model needs ~14GB)")
        raise

class LlamaFederatedClient:
    """Federated client for real LLaMA-2 model"""
    
    def __init__(self, client_id: int, model, tokenizer, config: Dict, device: str):
        self.client_id = client_id
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['optimizer']['client_lr'],
            weight_decay=config['training']['weight_decay'],
            eps=config['optimizer']['eps'],
            betas=config['optimizer']['betas']
        )
        
        # Training state
        self.current_round = 0
        self.local_updates = 0
        
        self.logger = logging.getLogger(f"LlamaClient-{client_id}")
        
    def train_local_epoch(self, dataloader):
        """Train one local epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # For causal LM, labels are the same as input_ids
            labels = input_ids.clone()
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Gradient accumulation
            accumulation_steps = self.config['training']['gradient_accumulation_steps']
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.local_updates += 1
            
            total_loss += loss.item() * accumulation_steps
            num_batches += 1
            
            if batch_idx % self.config['training']['logging_steps'] == 0:
                self.logger.info(f"Client {self.client_id}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'loss': avg_loss, 'num_batches': num_batches}
    
    def local_training_round(self, dataloader, server_comm_fn=None):
        """Perform local training round"""
        self.current_round += 1
        self.logger.info(f"Starting local training round {self.current_round}")
        
        round_stats = []
        local_epochs = self.config['training']['local_epochs']
        
        for epoch in range(local_epochs):
            epoch_stats = self.train_local_epoch(dataloader)
            round_stats.append(epoch_stats)
            
            self.logger.info(f"Epoch {epoch} completed. Loss: {epoch_stats['loss']:.4f}")
        
        # Compute round average
        avg_loss = np.mean([stats['loss'] for stats in round_stats])
        
        return {
            'round': self.current_round,
            'client_id': self.client_id,
            'avg_loss': avg_loss,
            'local_updates': self.local_updates
        }
    
    def get_model_parameters(self):
        """Get model parameters for federated averaging"""
        return {name: param.clone().cpu() for name, param in self.model.named_parameters()}
    
    def set_model_parameters(self, parameters):
        """Set model parameters from federated averaging"""
        for name, param in self.model.named_parameters():
            if name in parameters:
                param.data = parameters[name].to(param.device)
    
    def evaluate_local_model(self, dataloader):
        """Evaluate local model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = input_ids.clone()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'eval_loss': avg_loss,
            'perplexity': perplexity,
            'num_eval_samples': num_batches
        }
    
    def reset_for_new_round(self):
        """Reset for new round"""
        self.optimizer.zero_grad()

class LlamaClientManager:
    """Manager for LLaMA federated clients"""
    
    def __init__(self, base_model, tokenizer, client_configs, device):
        self.clients = []
        self.device = device
        
        # Create clients with copies of the base model
        for i, config in enumerate(client_configs):
            # Create a copy of the model for each client
            # Note: This is memory intensive for large models
            # In practice, you might want to use model sharding or other techniques
            client_model = type(base_model).from_pretrained(
                config['model']['model_name'],
                torch_dtype=torch.float16,
                **{k: v for k, v in base_model.config.__dict__.items() 
                   if k in ['load_in_8bit', 'load_in_4bit']}
            )
            
            if not (config.get('model', {}).get('load_in_8bit', False) or 
                    config.get('model', {}).get('load_in_4bit', False)):
                client_model = client_model.to(device)
            
            client = LlamaFederatedClient(i, client_model, tokenizer, config, device)
            self.clients.append(client)
        
        self.logger = logging.getLogger("LlamaClientManager")
        self.logger.info(f"Created {len(self.clients)} LLaMA federated clients")
    
    def get_all_clients(self):
        return self.clients
    
    def evaluate_all_clients(self, eval_dataloaders):
        """Evaluate all clients"""
        results = {}
        for i, (client, dataloader) in enumerate(zip(self.clients, eval_dataloaders)):
            results[i] = client.evaluate_local_model(dataloader)
        return results
    
    def save_all_clients(self, save_dir: str):
        """Save all client models"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for client in self.clients:
            client_dir = save_dir / f"client_{client.client_id}"
            client.model.save_pretrained(client_dir)
            # Save tokenizer too
            client.tokenizer.save_pretrained(client_dir)
        
        self.logger.info(f"Saved {len(self.clients)} clients to {save_dir}")

class FederatedLlamaTrainingCoordinator:
    """Training coordinator for federated LLaMA-2"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Set random seed
        set_random_seed(config.get('random_seed', 42))
        
        # Setup logging
        self.logger = setup_logging(
            log_level=config.get('logging', {}).get('log_level', 'INFO'),
            log_file=config.get('logging', {}).get('log_file', 'training.log')
        )
        
        # Initialize wandb
        if config.get('use_wandb', False):
            try:
                wandb.init(
                    project=config.get('logging', {}).get('wandb_project', 'llama2-federated-medical'),
                    config=config,
                    name=f"llama2_federated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize wandb: {e}")
        
        # Load LLaMA-2 model and tokenizer
        model_name = config['model']['model_name']
        self.base_model, self.tokenizer = load_llama2_model_and_tokenizer(
            model_name, self.device, config
        )
        
        self.logger.info("✅ Real LLaMA-2-7B model loaded successfully")
        self.logger.info("🔥 This is the actual 7 billion parameter LLaMA-2 model")
        
        # Training state
        self.current_round = 0
        self.best_metrics = {}
        
    def setup_data(self):
        """Setup federated data"""
        self.logger.info("Setting up federated data for LLaMA-2...")
        
        # Load medical QA data
        data_loader = MedicalQADataLoader(tokenizer_name=self.config['model']['model_name'])
        
        dataset = data_loader.create_dataset(
            dataset_name=self.config['data']['dataset_name'],
            max_samples=self.config['data']['combined_dataset_size'],
            max_length=self.config['model']['max_seq_length']
        )
        
        # Get statistics
        stats = data_loader.get_data_statistics(dataset)
        self.logger.info(f"Dataset loaded: {stats['num_samples']} samples")
        
        # Split dataset
        total_samples = len(dataset)
        train_size = int(total_samples * self.config['data']['train_split'])
        val_size = int(total_samples * self.config['data']['val_split'])
        test_size = total_samples - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create federated dataloaders
        num_clients = self.config['federated']['num_clients']
        batch_size = self.config['training']['batch_size']
        
        train_dataloaders = create_federated_dataloaders(
            train_dataset, num_clients, batch_size, 
            iid=self.config['data']['iid'], alpha=self.config['data']['alpha']
        )
        
        val_dataloaders = create_federated_dataloaders(
            val_dataset, num_clients, batch_size, iid=True
        )
        
        test_dataloaders = create_federated_dataloaders(
            test_dataset, num_clients, batch_size, iid=True
        )
        
        return train_dataloaders, val_dataloaders, test_dataloaders
    
    def setup_clients(self):
        """Setup federated clients"""
        self.logger.info("Setting up LLaMA-2 federated clients...")
        
        client_configs = [self.config.copy() for _ in range(self.config['federated']['num_clients'])]
        
        client_manager = LlamaClientManager(
            self.base_model, self.tokenizer, client_configs, self.device
        )
        
        return client_manager
    
    def federated_averaging(self, clients):
        """Perform federated averaging of model parameters"""
        self.logger.info("Performing federated averaging...")
        
        # Collect parameters from all clients
        all_parameters = [client.get_model_parameters() for client in clients]
        
        if not all_parameters:
            return
        
        # Average parameters
        averaged_params = {}
        for param_name in all_parameters[0].keys():
            # Stack parameters from all clients
            param_stack = torch.stack([params[param_name] for params in all_parameters])
            # Average
            averaged_params[param_name] = torch.mean(param_stack, dim=0)
        
        # Set averaged parameters to all clients
        for client in clients:
            client.set_model_parameters(averaged_params)
        
        self.logger.info("✅ Federated averaging completed")
    
    def run_training(self):
        """Run federated training"""
        self.logger.info("🚀 Starting federated LLaMA-2 training...")
        
        # Setup data and clients
        train_dataloaders, val_dataloaders, test_dataloaders = self.setup_data()
        client_manager = self.setup_clients()
        clients = client_manager.get_all_clients()
        
        # Training loop
        num_rounds = self.config['training']['num_rounds']
        
        for round_num in range(1, num_rounds + 1):
            self.logger.info(f"🔄 Starting federated round {round_num}/{num_rounds}")
            
            # Local training on each client
            round_stats = []
            for i, (client, dataloader) in enumerate(zip(clients, train_dataloaders)):
                self.logger.info(f"Training client {i}...")
                client_stats = client.local_training_round(dataloader)
                round_stats.append(client_stats)
            
            # Federated averaging
            if round_num % self.config.get('fedavg_frequency', 1) == 0:
                self.federated_averaging(clients)
            
            # Evaluation
            if round_num % self.config['evaluation']['eval_frequency'] == 0:
                eval_results = client_manager.evaluate_all_clients(val_dataloaders)
                
                avg_loss = np.mean([result['eval_loss'] for result in eval_results.values()])
                avg_perplexity = np.mean([result['perplexity'] for result in eval_results.values()])
                
                self.logger.info(f"Round {round_num} Evaluation:")
                self.logger.info(f"  Average Loss: {avg_loss:.4f}")
                self.logger.info(f"  Average Perplexity: {avg_perplexity:.2f}")
                
                # Log to wandb
                if self.config.get('use_wandb', False):
                    wandb.log({
                        'round': round_num,
                        'avg_eval_loss': avg_loss,
                        'avg_perplexity': avg_perplexity
                    })
                
                # Save best model
                if avg_loss < self.best_metrics.get('loss', float('inf')):
                    self.best_metrics['loss'] = avg_loss
                    self.best_metrics['round'] = round_num
                    
                    # Save best model
                    best_dir = Path('results/best_llama2_model')
                    best_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save the first client's model as the best model
                    clients[0].model.save_pretrained(best_dir)
                    self.tokenizer.save_pretrained(best_dir)
                    
                    self.logger.info(f"💾 New best model saved (loss: {avg_loss:.4f})")
            
            # Save checkpoint
            if round_num % 5 == 0:
                checkpoint_dir = Path(f'results/checkpoints/round_{round_num}')
                client_manager.save_all_clients(str(checkpoint_dir))
                self.logger.info(f"💾 Checkpoint saved at round {round_num}")
        
        # Final evaluation
        self.logger.info("🏁 Running final evaluation...")
        final_eval = client_manager.evaluate_all_clients(test_dataloaders)
        
        final_loss = np.mean([result['eval_loss'] for result in final_eval.values()])
        final_perplexity = np.mean([result['perplexity'] for result in final_eval.values()])
        
        self.logger.info("✅ Training completed!")
        self.logger.info(f"📊 Final Results:")
        self.logger.info(f"  Test Loss: {final_loss:.4f}")
        self.logger.info(f"  Test Perplexity: {final_perplexity:.2f}")
        self.logger.info(f"  Best Round: {self.best_metrics.get('round', 'N/A')}")
        self.logger.info(f"  Best Loss: {self.best_metrics.get('loss', 'N/A'):.4f}")
        
        return {
            'final_test_loss': final_loss,
            'final_test_perplexity': final_perplexity,
            'best_metrics': self.best_metrics,
            'model_info': {
                'model_name': self.config['model']['model_name'],
                'total_parameters': sum(p.numel() for p in self.base_model.parameters()),
                'is_real_llama2': True
            }
        }

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Federated LLaMA-2-7B Training for Medical QA')
    parser.add_argument('--config', type=str, default='config/llama2_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--num_clients', type=int, default=None,
                       help='Number of federated clients')
    parser.add_argument('--num_rounds', type=int, default=None,
                       help='Number of training rounds')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--dataset_size', type=int, default=None,
                       help='Maximum dataset size')
    parser.add_argument('--load_in_8bit', action='store_true',
                       help='Load model in 8-bit quantization')
    parser.add_argument('--load_in_4bit', action='store_true',
                       help='Load model in 4-bit quantization')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size per client')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        if Path(args.config).exists():
            config = load_and_validate_config(args.config)
            print(f"✓ Loaded configuration from: {args.config}")
        else:
            print(f"⚠️  Configuration file not found: {args.config}")
            print("Using default configuration...")
            config = create_default_config()
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        print("Using default configuration...")
        config = create_default_config()
    
    # Override config with command line arguments
    if args.device:
        config['device'] = args.device
    if args.num_clients:
        config['federated']['num_clients'] = args.num_clients
    if args.num_rounds:
        config['training']['num_rounds'] = args.num_rounds
    if args.use_wandb:
        config['use_wandb'] = True
    if args.dataset_size:
        config['data']['combined_dataset_size'] = args.dataset_size
    if args.load_in_8bit:
        config['model']['load_in_8bit'] = True
        print("📉 Enabled 8-bit quantization")
    if args.load_in_4bit:
        config['model']['load_in_4bit'] = True
        print("📉 Enabled 4-bit quantization")
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    print(f"🦙 LLaMA-2-7B Federated Learning Configuration:")
    print(f"  Model: {config['model']['model_name']}")
    print(f"  Clients: {config['federated']['num_clients']}")
    print(f"  Rounds: {config['training']['num_rounds']}")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Learning Rate: {config['optimizer']['client_lr']}")
    print(f"  Dataset Size: {config['data']['combined_dataset_size']}")
    print(f"  8-bit Loading: {config['model'].get('load_in_8bit', False)}")
    print(f"  4-bit Loading: {config['model'].get('load_in_4bit', False)}")
    print(f"  🔥 Using REAL LLaMA-2-7B Model (7 billion parameters)")
    
    # Create results directories
    Path('results/logs').mkdir(parents=True, exist_ok=True)
    Path('results/checkpoints').mkdir(parents=True, exist_ok=True)
    Path('results/best_llama2_model').mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize training coordinator
        coordinator = FederatedLlamaTrainingCoordinator(config)
        
        # Run training
        training_summary = coordinator.run_training()
        
        print("🎉 LLaMA-2 Federated Training completed successfully!")
        print(f"📊 Final Results:")
        print(f"  Test Loss: {training_summary['final_test_loss']:.4f}")
        print(f"  Test Perplexity: {training_summary['final_test_perplexity']:.2f}")
        print(f"  Model: {training_summary['model_info']['model_name']}")
        print(f"  Total Parameters: {training_summary['model_info']['total_parameters']:,}")
        print(f"  Real LLaMA-2: {training_summary['model_info']['is_real_llama2']}")
        
        # Save final results
        results_dir = Path('results/final')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = results_dir / f"llama2_federated_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(results_path, 'w') as f:
            yaml.dump(training_summary, f, default_flow_style=False)
        
        print(f"💾 Results saved to: {results_path}")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Check for common issues
        if "access" in str(e).lower() or "permission" in str(e).lower():
            print("\n💡 Possible solutions:")
            print("1. Request access to LLaMA-2 on Hugging Face Hub:")
            print("   https://huggingface.co/meta-llama/Llama-2-7b-hf")
            print("2. Login with: huggingface-cli login")
        elif "memory" in str(e).lower() or "cuda" in str(e).lower():
            print("\n💡 Memory optimization suggestions:")
            print("1. Use --load_in_8bit or --load_in_4bit flags")
            print("2. Reduce --batch_size (try 1)")
            print("3. Reduce --num_clients")
            print("4. Use gradient checkpointing")
        
        raise
    
    finally:
        # Clean up
        if config.get('use_wandb', False):
            try:
                wandb.finish()
            except:
                pass

if __name__ == "__main__":
    main()
