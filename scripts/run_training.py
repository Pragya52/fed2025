#!/usr/bin/env python3
"""
Main Training Script for Federated Split Learning with Medical QA
Implements the complete training pipeline from the methodology
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
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.llama_split import create_llama_models, load_pretrained_weights
from src.federated.client import ClientManager
from src.federated.server import FederatedServer, ServerCommunicationHandler
from src.data.medical_qa_loader import MedicalQADataLoader, create_federated_dataloaders
from src.training.metrics import MedicalQAMetrics, FederatedMetricsAggregator
from src.utils.logger import setup_logging
from src.utils.helpers import set_random_seed, save_checkpoint, load_checkpoint

class FederatedTrainingCoordinator:
    """
    Coordinates federated training across clients and server
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Set random seed for reproducibility
        set_random_seed(config.get('random_seed', 42))
        
        # Initialize logging
        self.logger = setup_logging(
            log_level=config['logging']['log_level'],
            log_file=config['logging']['log_file']
        )
        
        # Initialize wandb if configured
        if config.get('use_wandb', False):
            wandb.init(
                project=config['logging']['wandb_project'],
                config=config,
                name=f"federated_medical_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Create models
        self.client_model, self.server_model = create_llama_models(config['model'])
        
        # Load pretrained weights if specified
        if config['model'].get('pretrained_model_name'):
            load_pretrained_weights(
                self.client_model, 
                self.server_model, 
                config['model']['pretrained_model_name']
            )
        
        # Create federated components
        self.server = FederatedServer(self.server_model, config, self.device)
        self.server_comm_handler = ServerCommunicationHandler(self.server)
        
        # Initialize metrics
        self.metrics_aggregator = FederatedMetricsAggregator()
        
        # Training state
        self.current_round = 0
        self.best_metrics = {}
        
        self.logger.info("Federated Training Coordinator initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Number of clients: {config['federated']['num_clients']}")
    
    def setup_data(self) -> Tuple[List, List, List]:
        """
        Setup federated data for training and evaluation
        
        Returns:
            Tuple of (train_dataloaders, val_dataloaders, test_dataloaders)
        """
        self.logger.info("Setting up federated data...")
        
        # Initialize data loader
        data_loader = MedicalQADataLoader(
            tokenizer_name=self.config['model'].get('tokenizer_name', 'microsoft/DialoGPT-medium')
        )
        
        # Create dataset
        dataset = data_loader.create_dataset(
            dataset_name=self.config['data'].get('dataset_name', 'sample'),
            max_length=self.config['model']['max_seq_length'],
            task_type='qa'
        )
        
        # Get dataset statistics
        stats = data_loader.get_data_statistics(dataset)
        self.logger.info(f"Dataset loaded: {stats['num_samples']} samples")
        self.logger.info(f"Average question length: {stats['avg_question_length']:.1f}")
        self.logger.info(f"Average answer length: {stats['avg_answer_length']:.1f}")
        
        # Split data for train/val/test
        total_samples = len(dataset)
        train_size = int(total_samples * self.config['data']['train_split'])
        val_size = int(total_samples * self.config['data']['val_split'])
        test_size = total_samples - train_size - val_size
        
        # Random split
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create federated dataloaders
        train_dataloaders = create_federated_dataloaders(
            train_dataset,
            num_clients=self.config['federated']['num_clients'],
            batch_size=self.config['training']['batch_size'],
            iid=self.config['data']['iid'],
            alpha=self.config['data']['alpha']
        )
        
        val_dataloaders = create_federated_dataloaders(
            val_dataset,
            num_clients=self.config['federated']['num_clients'],
            batch_size=self.config['training']['batch_size'],
            iid=True  # Use IID for validation
        )
        
        test_dataloaders = create_federated_dataloaders(
            test_dataset,
            num_clients=self.config['federated']['num_clients'],
            batch_size=self.config['training']['batch_size'],
            iid=True  # Use IID for testing
        )
        
        # Log data distribution
        for i, loader in enumerate(train_dataloaders):
            self.logger.info(f"Client {i}: {len(loader.dataset)} training samples")
        
        return train_dataloaders, val_dataloaders, test_dataloaders
    
    def setup_clients(self, train_dataloaders: List) -> ClientManager:
        """
        Setup federated clients
        
        Args:
            train_dataloaders: List of training dataloaders for each client
            
        Returns:
            ClientManager instance
        """
        self.logger.info("Setting up federated clients...")
        
        # Create client configurations
        client_configs = []
        for i in range(self.config['federated']['num_clients']):
            client_config = self.config.copy()
            client_configs.append(client_config)
        
        # Model factory function
        def model_factory(config):
            client_model, _ = create_llama_models(config['model'])
            return client_model
        
        # Create client manager
        client_manager = ClientManager(
            client_configs=client_configs,
            model_factory=model_factory,
            device=self.device
        )
        
        # Register clients with server
        for client in client_manager.get_all_clients():
            self.server.register_client(client.client_id)
        
        self.logger.info(f"Created {len(client_manager.get_all_clients())} federated clients")
        
        return client_manager
    
    def run_federated_round(self, 
                           round_num: int,
                           client_manager: ClientManager,
                           train_dataloaders: List,
                           val_dataloaders: List) -> Dict[str, Any]:
        """
        Run one complete federated training round
        
        Args:
            round_num: Current round number
            client_manager: Client manager
            train_dataloaders: Training dataloaders
            val_dataloaders: Validation dataloaders
            
        Returns:
            Round statistics
        """
        self.logger.info(f"Starting federated round {round_num}")
        
        # Reset clients for new round
        for client in client_manager.get_all_clients():
            client.reset_for_new_round()
        
        # Run coordinated federated round
        round_stats = self.server_comm_handler.coordinate_federated_round(
            client_managers=client_manager.get_all_clients(),
            client_dataloaders=train_dataloaders
        )
        
        # Evaluate clients if it's evaluation round
        if round_num % self.config['evaluation']['eval_frequency'] == 0:
            self.logger.info("Running evaluation...")
            
            # Client evaluation
            client_eval_results = client_manager.evaluate_all_clients(val_dataloaders)
            
            # Add evaluation metrics to round stats
            for client_id, eval_metrics in client_eval_results.items():
                self.metrics_aggregator.add_client_metrics(client_id, eval_metrics)
            
            # Server evaluation (if we have validation data)
            # This would require implementing server evaluation with proper data format
            
            round_stats['client_evaluations'] = client_eval_results
        
        # Add server metrics to aggregator
        server_metrics = {k: v for k, v in round_stats.items() 
                         if isinstance(v, (int, float))}
        self.metrics_aggregator.add_server_metrics(server_metrics)
        
        # Log round statistics
        self.logger.info(f"Round {round_num} completed:")
        self.logger.info(f"  Successful clients: {round_stats.get('successful_clients', 0)}")
        self.logger.info(f"  Global loss: {round_stats.get('total_loss', 0):.4f}")
        
        # Log to wandb if enabled
        if self.config.get('use_wandb', False):
            wandb_metrics = {f"round_{k}": v for k, v in round_stats.items() 
                           if isinstance(v, (int, float))}
            wandb_metrics['round'] = round_num
            wandb.log(wandb_metrics, step=round_num)
        
        return round_stats
    
    def save_training_checkpoint(self, 
                               round_num: int,
                               client_manager: ClientManager,
                               round_stats: Dict[str, Any]):
        """Save training checkpoint"""
        if round_num % self.config['logging']['save_frequency'] == 0:
            checkpoint_dir = Path(self.config['logging'].get('checkpoint_dir', 'results/checkpoints'))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save server state
            server_path = checkpoint_dir / f"server_round_{round_num}.pt"
            torch.save(self.server.get_model_state(), server_path)
            
            # Save client states
            client_dir = checkpoint_dir / f"clients_round_{round_num}"
            client_manager.save_all_clients(str(client_dir))
            
            # Save training metadata
            metadata = {
                'round': round_num,
                'config': self.config,
                'round_stats': round_stats,
                'metrics_summary': self.metrics_aggregator.get_training_summary()
            }
            
            metadata_path = checkpoint_dir / f"metadata_round_{round_num}.yaml"
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f)
            
            self.logger.info(f"Saved checkpoint for round {round_num}")
    
    def run_training(self):
        """Run complete federated training"""
        self.logger.info("Starting federated split learning training...")
        
        # Setup data and clients
        train_dataloaders, val_dataloaders, test_dataloaders = self.setup_data()
        client_manager = self.setup_clients(train_dataloaders)
        
        # Training loop
        training_history = []
        
        for round_num in range(1, self.config['training']['num_rounds'] + 1):
            try:
                # Check if training should continue (privacy budget, etc.)
                if not self.server.should_continue_training():
                    self.logger.warning("Training stopped due to constraints (e.g., privacy budget)")
                    break
                
                # Run federated round
                round_stats = self.run_federated_round(
                    round_num, client_manager, train_dataloaders, val_dataloaders
                )
                
                training_history.append(round_stats)
                
                # Save checkpoint
                self.save_training_checkpoint(round_num, client_manager, round_stats)
                
                # Check for best model
                if 'client_evaluations' in round_stats:
                    avg_accuracy = np.mean([
                        metrics.get('accuracy', 0) 
                        for metrics in round_stats['client_evaluations'].values()
                    ])
                    
                    if avg_accuracy > self.best_metrics.get('accuracy', 0):
                        self.best_metrics['accuracy'] = avg_accuracy
                        self.best_metrics['round'] = round_num
                        
                        # Save best model
                        best_model_dir = Path('results/best_model')
                        best_model_dir.mkdir(parents=True, exist_ok=True)
                        
                        torch.save(self.server.get_model_state(), 
                                 best_model_dir / 'best_server_model.pt')
                        client_manager.save_all_clients(str(best_model_dir / 'best_clients'))
                        
                        self.logger.info(f"New best model saved (accuracy: {avg_accuracy:.4f})")
                
            except Exception as e:
                self.logger.error(f"Error in round {round_num}: {e}")
                continue
        
        # Final evaluation
        self.logger.info("Running final evaluation...")
        final_eval_results = self.run_final_evaluation(
            client_manager, test_dataloaders
        )
        
        # Generate training summary
        training_summary = self.generate_training_summary(
            training_history, final_eval_results
        )
        
        # Save final results
        self.save_final_results(training_summary)
        
        self.logger.info("Federated training completed!")
        
        return training_summary
    
    def run_final_evaluation(self, 
                           client_manager: ClientManager,
                           test_dataloaders: List) -> Dict[str, Any]:
        """Run final evaluation on test data"""
        self.logger.info("Running final evaluation on test data...")
        
        # Evaluate all clients on test data
        test_results = client_manager.evaluate_all_clients(test_dataloaders)
        
        # Compute federated averages
        for client_id, metrics in test_results.items():
            self.metrics_aggregator.add_client_metrics(client_id, metrics)
        
        federated_averages = self.metrics_aggregator.compute_federated_averages()
        fairness_metrics = self.metrics_aggregator.compute_client_fairness_metrics()
        
        final_results = {
            'client_results': test_results,
            'federated_averages': federated_averages,
            'fairness_metrics': fairness_metrics,
            'privacy_analysis': self.server.privacy_accountant.get_privacy_analysis(),
            'communication_stats': self.server.get_communication_statistics()
        }
        
        # Log key results
        self.logger.info("Final Evaluation Results:")
        self.logger.info(f"  Average Accuracy: {federated_averages.get('fed_avg_accuracy', 0):.4f}")
        self.logger.info(f"  Average F1: {federated_averages.get('fed_avg_f1', 0):.4f}")
        self.logger.info(f"  Privacy Budget Used: {final_results['privacy_analysis']['consumed_epsilon_basic']:.4f}")
        
        return final_results
    
    def generate_training_summary(self, 
                                training_history: List[Dict],
                                final_eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive training summary"""
        summary = {
            'training_config': self.config,
            'training_history': training_history,
            'final_evaluation': final_eval_results,
            'best_model_info': self.best_metrics,
            'server_summary': self.server.get_training_summary(),
            'client_summary': client_manager.get_training_statistics() if 'client_manager' in locals() else {},
            'total_rounds': len(training_history),
            'training_completed': datetime.now().isoformat()
        }
        
        return summary
    
    def save_final_results(self, training_summary: Dict[str, Any]):
        """Save final training results"""
        results_dir = Path('results/final')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive summary
        summary_path = results_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(summary_path, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_summary = self._make_serializable(training_summary)
            yaml.dump(serializable_summary, f, default_flow_style=False)
        
        # Save metrics in separate file for easier analysis
        metrics_path = results_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(metrics_path, 'w') as f:
            metrics_only = {
                'final_metrics': training_summary['final_evaluation']['federated_averages'],
                'fairness_metrics': training_summary['final_evaluation']['fairness_metrics'],
                'privacy_analysis': training_summary['final_evaluation']['privacy_analysis'],
                'communication_stats': training_summary['final_evaluation']['communication_stats']
            }
            yaml.dump(self._make_serializable(metrics_only), f, default_flow_style=False)
        
        self.logger.info(f"Final results saved to {results_dir}")
    
    def _make_serializable(self, obj):
        """Convert objects to YAML-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Federated Split Learning for Medical QA')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--num_clients', type=int, default=None,
                       help='Number of federated clients')
    parser.add_argument('--num_rounds', type=int, default=None,
                       help='Number of training rounds')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.device:
        config['device'] = args.device
    if args.num_clients:
        config['federated']['num_clients'] = args.num_clients
    if args.num_rounds:
        config['training']['num_rounds'] = args.num_rounds
    if args.use_wandb:
        config['use_wandb'] = True
    
    # Create results directories
    Path('results/logs').mkdir(parents=True, exist_ok=True)
    Path('results/checkpoints').mkdir(parents=True, exist_ok=True)
    Path('results/models').mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize training coordinator
        coordinator = FederatedTrainingCoordinator(config)
        
        # Resume from checkpoint if specified
        if args.resume_from:
            # Implementation for resuming from checkpoint
            # This would load server and client states
            coordinator.logger.info(f"Resuming training from {args.resume_from}")
            # TODO: Implement checkpoint loading
        
        # Run training
        training_summary = coordinator.run_training()
        
        print("Training completed successfully!")
        print(f"Final accuracy: {training_summary['final_evaluation']['federated_averages'].get('fed_avg_accuracy', 0):.4f}")
        print(f"Privacy budget used: {training_summary['final_evaluation']['privacy_analysis']['consumed_epsilon_basic']:.4f}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    finally:
        # Clean up
        if config.get('use_wandb', False):
            wandb.finish()

if __name__ == "__main__":
    main()
