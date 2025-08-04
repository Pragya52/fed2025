"""
Federated Client Implementation
Handles local training, privacy mechanisms, and server communication
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging
from copy import deepcopy

from ..models.llama_split import LLaMAClientModel
from ..models.privacy_mechanisms import PrivacyPreservingTransmission
from ..training.losses import FederatedKnowledgeDistillationLoss
from ..training.metrics import MedicalQAMetrics

class FederatedClient:
    """
    Federated Learning Client for Medical QA
    Implements the client-side logic from the methodology
    """
    
    def __init__(self, 
                 client_id: int,
                 model: LLaMAClientModel,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        """
        Args:
            client_id: Unique identifier for this client
            model: Client model instance
            config: Configuration dictionary
            device: Device to run computations on
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Privacy mechanisms
        self.privacy_mechanism = PrivacyPreservingTransmission(
            sigma=config['privacy']['gaussian_noise']['sigma'],
            num_bits=config['privacy']['quantization']['bits'],
            device=device
        )
        
        # Loss functions
        self.loss_fn = FederatedKnowledgeDistillationLoss(
            temperature=config['training']['temperature'],
            alpha=config['loss_weights']['client']['mu1'],
            beta=config['loss_weights']['client']['mu2']
        )
        
        # Metrics
        self.metrics = MedicalQAMetrics()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Training state
        self.current_round = 0
        self.local_updates = 0
        self.training_history = []
        
        # Logger
        self.logger = logging.getLogger(f"Client-{client_id}")
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for local training"""
        optimizer_config = self.config['optimizer']
        
        if optimizer_config['name'] == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config['client_lr'],
                weight_decay=self.config['training']['weight_decay'],
                eps=optimizer_config['eps']
            )
        elif optimizer_config['name'] == 'SGD':
            return optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['client_lr'],
                momentum=optimizer_config['momentum'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")
    
    def local_training_step(self, 
                           batch: Dict[str, torch.Tensor],
                           server_predictions: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Perform one local training step
        
        Args:
            batch: Training batch
            server_predictions: Predictions from server (for distillation)
            
        Returns:
            Dictionary with loss values and metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass through local model
        local_logits = self.model.forward_local(input_ids, attention_mask)
        
        # Compute local predictions
        local_predictions = torch.softmax(local_logits, dim=-1)
        
        # Prepare server predictions if available
        if server_predictions is not None:
            server_predictions = server_predictions.to(self.device)
        
        # Compute loss
        loss_dict = self.loss_fn(
            client_logits=local_logits,
            server_logits=server_predictions,
            labels=labels,
            client_predictions=local_predictions,
            server_predictions=server_predictions
        )
        
        total_loss = loss_dict['total_loss']
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if self.config['training']['gradient_clip_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['gradient_clip_norm']
            )
        
        # Optimizer step
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            metrics = self.metrics.compute_batch_metrics(
                predictions=local_predictions,
                labels=labels,
                input_ids=input_ids
            )
        
        # Combine loss and metrics
        result = {**loss_dict, **metrics}
        
        return result
    
    def prepare_smashed_data(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Prepare smashed data for transmission to server
        
        Args:
            batch: Input batch
            
        Returns:
            Tuple of (quantized_smashed_data, scaling_parameters)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass through client-side global model path
            smashed_data = self.model.forward_global_client_side(input_ids, attention_mask)
            
            # Apply privacy mechanisms
            quantized_data, scaling_params = self.privacy_mechanism.prepare_transmission(smashed_data)
        
        return quantized_data, scaling_params
    
    def local_training_round(self, 
                           dataloader: DataLoader,
                           server_communication_fn: callable) -> Dict[str, float]:
        """
        Perform one complete local training round
        
        Args:
            dataloader: Local training data
            server_communication_fn: Function to communicate with server
            
        Returns:
            Round training statistics
        """
        self.current_round += 1
        round_stats = {'losses': [], 'metrics': []}
        
        self.logger.info(f"Starting local training round {self.current_round}")
        
        for epoch in range(self.config['training']['local_epochs']):
            epoch_stats = {'losses': [], 'metrics': []}
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Step 1: Prepare smashed data for server
                    quantized_data, scaling_params = self.prepare_smashed_data(batch)
                    
                    # Step 2: Send smashed data to server and receive predictions
                    server_predictions = server_communication_fn({
                        'client_id': self.client_id,
                        'smashed_data': quantized_data,
                        'scaling_params': scaling_params,
                        'labels': batch['labels'],
                        'attention_mask': batch['attention_mask']
                    })
                    
                    # Step 3: Perform local training step with server feedback
                    step_results = self.local_training_step(batch, server_predictions)
                    
                    epoch_stats['losses'].append(step_results)
                    epoch_stats['metrics'].append(step_results)
                    
                    self.local_updates += 1
                    
                    if batch_idx % 10 == 0:
                        self.logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Loss: {step_results['total_loss']:.4f}")
                
                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx}: {e}")
                    continue
            
            # Compute epoch averages
            if epoch_stats['losses']:
                epoch_avg = self._compute_average_stats(epoch_stats['losses'])
                round_stats['losses'].extend(epoch_stats['losses'])
                round_stats['metrics'].extend(epoch_stats['metrics'])
                
                self.logger.info(f"Epoch {epoch} completed. Avg Loss: {epoch_avg['total_loss']:.4f}")
        
        # Compute round statistics
        round_summary = self._compute_round_summary(round_stats)
        self.training_history.append(round_summary)
        
        self.logger.info(f"Round {self.current_round} completed. Avg Loss: {round_summary['avg_total_loss']:.4f}")
        
        return round_summary
    
    def _compute_average_stats(self, stats_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute average of statistics"""
        if not stats_list:
            return {}
        
        avg_stats = {}
        for key in stats_list[0].keys():
            if isinstance(stats_list[0][key], (int, float)):
                avg_stats[key] = np.mean([stats[key] for stats in stats_list])
        
        return avg_stats
    
    def _compute_round_summary(self, round_stats: Dict[str, List]) -> Dict[str, float]:
        """Compute summary statistics for the round"""
        summary = {
            'round': self.current_round,
            'client_id': self.client_id,
            'num_batches': len(round_stats['losses']),
            'local_updates': self.local_updates
        }
        
        if round_stats['losses']:
            avg_losses = self._compute_average_stats(round_stats['losses'])
            for key, value in avg_losses.items():
                summary[f'avg_{key}'] = value
        
        return summary
    
    def get_head_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get head parameters for federated averaging
        
        Returns:
            Dictionary of head parameters
        """
        return self.model.get_head_parameters()
    
    def set_head_parameters(self, head_params: Dict[str, torch.Tensor]):
        """
        Set head parameters from federated averaging
        
        Args:
            head_params: New head parameters
        """
        self.model.set_head_parameters(head_params)
        self.logger.info("Updated head parameters from federated averaging")
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get complete model state for checkpointing"""
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_round': self.current_round,
            'local_updates': self.local_updates,
            'training_history': self.training_history
        }
    
    def load_model_state(self, state: Dict[str, Any]):
        """Load model state from checkpoint"""
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.current_round = state['current_round']
        self.local_updates = state['local_updates']
        self.training_history = state['training_history']
        
        self.logger.info(f"Loaded model state from round {self.current_round}")
    
    def evaluate_local_model(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate local model performance
        
        Args:
            dataloader: Evaluation data
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                local_logits = self.model.forward_local(input_ids, attention_mask)
                local_predictions = torch.softmax(local_logits, dim=-1)
                
                # Compute loss (without server predictions for evaluation)
                loss_dict = self.loss_fn(
                    client_logits=local_logits,
                    server_logits=None,  # No server predictions during evaluation
                    labels=labels,
                    client_predictions=local_predictions,
                    server_predictions=None
                )
                
                all_predictions.append(local_predictions)
                all_labels.append(labels)
                all_losses.append(loss_dict['task_loss'])  # Use task loss for evaluation
        
        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute comprehensive metrics
        metrics = self.metrics.compute_comprehensive_metrics(
            predictions=all_predictions,
            labels=all_labels,
            input_ids=None  # Not needed for evaluation metrics
        )
        
        # Add average loss
        metrics['eval_loss'] = torch.stack(all_losses).mean().item()
        
        self.logger.info(f"Client {self.client_id} evaluation completed. Loss: {metrics['eval_loss']:.4f}")
        
        return metrics
    
    def get_privacy_analysis(self) -> Dict[str, float]:
        """Get privacy analysis for this client"""
        # Create sample data for analysis
        sample_data = torch.randn(1, 10, self.config['model']['embed_dim'], device=self.device)
        
        # Prepare transmission
        quantized_data, scaling_params = self.privacy_mechanism.prepare_transmission(sample_data)
        reconstructed_data = self.privacy_mechanism.reconstruct_transmission(quantized_data, scaling_params)
        
        # Analyze privacy-utility tradeoff
        analysis = self.privacy_mechanism.analyze_privacy_utility_tradeoff(
            sample_data, reconstructed_data, delta=self.config['privacy']['differential_privacy']['delta']
        )
        
        return analysis
    
    def reset_for_new_round(self):
        """Reset client state for new federated round"""
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Update learning rate if scheduler is used
        # This can be extended to include learning rate scheduling
        
        self.logger.debug(f"Client {self.client_id} reset for new round")

class ClientManager:
    """Manages multiple federated clients"""
    
    def __init__(self, 
                 client_configs: List[Dict[str, Any]],
                 model_factory: callable,
                 device: str = 'cuda'):
        """
        Args:
            client_configs: List of configuration dictionaries for each client
            model_factory: Function to create client models
            device: Device to run computations on
        """
        self.clients = []
        self.device = device
        
        # Create clients
        for i, config in enumerate(client_configs):
            model = model_factory(config)
            client = FederatedClient(
                client_id=i,
                model=model,
                config=config,
                device=device
            )
            self.clients.append(client)
        
        self.logger = logging.getLogger("ClientManager")
        self.logger.info(f"Created {len(self.clients)} federated clients")
    
    def get_client(self, client_id: int) -> FederatedClient:
        """Get client by ID"""
        if 0 <= client_id < len(self.clients):
            return self.clients[client_id]
        else:
            raise ValueError(f"Invalid client ID: {client_id}")
    
    def get_all_clients(self) -> List[FederatedClient]:
        """Get all clients"""
        return self.clients
    
    def collect_head_parameters(self) -> List[Dict[str, torch.Tensor]]:
        """Collect head parameters from all clients for federated averaging"""
        head_params = []
        for client in self.clients:
            head_params.append(client.get_head_parameters())
        return head_params
    
    def distribute_head_parameters(self, aggregated_params: Dict[str, torch.Tensor]):
        """Distribute aggregated head parameters to all clients"""
        for client in self.clients:
            client.set_head_parameters(aggregated_params)
    
    def evaluate_all_clients(self, eval_dataloaders: List[DataLoader]) -> Dict[int, Dict[str, float]]:
        """Evaluate all clients on their respective datasets"""
        results = {}
        
        for i, (client, dataloader) in enumerate(zip(self.clients, eval_dataloaders)):
            results[i] = client.evaluate_local_model(dataloader)
        
        return results
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics from all clients"""
        stats = {
            'num_clients': len(self.clients),
            'total_local_updates': sum(client.local_updates for client in self.clients),
            'current_rounds': [client.current_round for client in self.clients],
            'client_histories': [client.training_history for client in self.clients]
        }
        
        return stats
    
    def save_all_clients(self, save_dir: str):
        """Save state of all clients"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for client in self.clients:
            save_path = os.path.join(save_dir, f"client_{client.client_id}.pt")
            torch.save(client.get_model_state(), save_path)
        
        self.logger.info(f"Saved {len(self.clients)} clients to {save_dir}")
    
    def load_all_clients(self, save_dir: str):
        """Load state of all clients"""
        import os
        
        for client in self.clients:
            save_path = os.path.join(save_dir, f"client_{client.client_id}.pt")
            if os.path.exists(save_path):
                state = torch.load(save_path, map_location=self.device)
                client.load_model_state(state)
        
        self.logger.info(f"Loaded clients from {save_dir}")

# Test function
def test_federated_client():
    """Test the federated client implementation"""
    from ..models.llama_split import create_llama_models
    
    # Mock configuration
    config = {
        'model': {'embed_dim': 256, 'vocab_size': 1000, 'max_seq_length': 128},
        'privacy': {
            'gaussian_noise': {'sigma': 0.1},
            'quantization': {'bits': 8},
            'differential_privacy': {'delta': 1e-5}
        },
        'training': {
            'temperature': 3.0,
            'local_epochs': 2,
            'gradient_clip_norm': 1.0,
            'weight_decay': 0.01
        },
        'loss_weights': {
            'client': {'mu1': 0.5, 'mu2': 0.5}
        },
        'optimizer': {
            'name': 'AdamW',
            'client_lr': 1e-4,
            'eps': 1e-8
        }
    }
    
    # Create model
    client_model, _ = create_llama_models(config['model'])
    
    # Create client
    client = FederatedClient(
        client_id=0,
        model=client_model,
        config=config,
        device='cpu'  # Use CPU for testing
    )
    
    # Test privacy analysis
    privacy_analysis = client.get_privacy_analysis()
    print("Privacy Analysis:")
    for key, value in privacy_analysis.items():
        print(f"  {key}: {value:.6f}")
    
    # Test parameter operations
    head_params = client.get_head_parameters()
    print(f"\nHead parameters: {len(head_params)} tensors")
    
    client.set_head_parameters(head_params)  # Test setting same parameters
    
    print("Federated client test completed successfully!")

if __name__ == "__main__":
    test_federated_client()
