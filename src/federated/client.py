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
from ..training.losses import MedicalFederatedDistillationLoss  # Fixed import
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
        privacy_config = config.get('privacy', {})
        gaussian_noise = privacy_config.get('gaussian_noise', {})
        quantization = privacy_config.get('quantization', {})
        
        self.privacy_mechanism = PrivacyPreservingTransmission(
            sigma=gaussian_noise.get('sigma', 0.1),
            num_bits=quantization.get('bits', 8),
            device=device
        )
        
        # Loss functions - Updated to use correct class and parameter structure
        loss_config = config.get('loss', {})
        self.loss_fn = MedicalFederatedDistillationLoss(
            temperature=loss_config.get('temperature', 3.0),
            alpha=loss_config.get('distillation_weight', 0.5),
            beta=loss_config.get('task_weight', 0.5),
            gamma=loss_config.get('medical_weight', 0.1),
            label_smoothing=loss_config.get('label_smoothing', 0.1),
            medical_term_boost=loss_config.get('medical_term_boost', 2.0)
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
        
        # Ensure learning rate is a float (handle string values from YAML)
        client_lr = optimizer_config.get('client_lr', 5e-5)
        if isinstance(client_lr, str):
            client_lr = float(client_lr)
        
        # Ensure weight decay is a float
        weight_decay = self.config.get('training', {}).get('weight_decay', 0.01)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)
        
        # Ensure eps is a float
        eps = optimizer_config.get('eps', 1e-8)
        if isinstance(eps, str):
            eps = float(eps)
        
        if optimizer_config.get('name', 'AdamW') == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=client_lr,
                weight_decay=weight_decay,
                eps=eps
            )
        elif optimizer_config.get('name', 'AdamW') == 'SGD':
            # Ensure momentum is a float
            momentum = optimizer_config.get('momentum', 0.9)
            if isinstance(momentum, str):
                momentum = float(momentum)
                
            return optim.SGD(
                self.model.parameters(),
                lr=client_lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config.get('name', 'AdamW')}")
    
    def _create_simple_medical_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Create a simple medical mask based on token patterns
        This is a placeholder - in practice, you'd use medical vocabulary
        """
        # Simple heuristic: assume some tokens are medical terms
        # This should be replaced with actual medical vocabulary matching
        medical_mask = torch.zeros_like(labels, dtype=torch.bool)
        
        # For now, mark first few tokens as potentially medical
        # In practice, you'd use a medical vocabulary lookup
        seq_len = labels.size(-1)
        medical_tokens = min(3, seq_len)  # First 3 tokens or sequence length, whichever is smaller
        medical_mask[:, :medical_tokens] = True
        
        return medical_mask
    
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
        
        # Prepare server predictions if available
        if server_predictions is not None:
            server_predictions = server_predictions.to(self.device)
        
        # Create medical mask (simple heuristic - can be improved)
        medical_mask = self._create_simple_medical_mask(labels)
        
        # Compute loss using the updated loss function signature
        loss_dict = self.loss_fn(
            client_logits=local_logits,
            server_logits=server_predictions,
            labels=labels,
            attention_mask=attention_mask,
            medical_mask=medical_mask
        )
        
        total_loss = loss_dict['total_loss']
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        gradient_clip_norm = self.config.get('training', {}).get('gradient_clip_norm', 1.0)
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                gradient_clip_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            # Compute local predictions for metrics
            local_predictions = torch.softmax(local_logits, dim=-1)
            
            metrics = self.metrics.compute_batch_metrics(
                predictions=local_predictions,
                labels=labels,
                input_ids=input_ids
            )
        
        # Combine loss and metrics (convert tensors to floats)
        result = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.item()
            else:
                result[key] = value
        
        # Add metrics
        result.update(metrics)
        
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
        
        local_epochs = self.config.get('training', {}).get('local_epochs', 2)
        
        for epoch in range(local_epochs):
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
                
                # Create medical mask for evaluation
                medical_mask = self._create_simple_medical_mask(labels)
                
                # Compute loss (without server predictions for evaluation)
                loss_dict = self.loss_fn(
                    client_logits=local_logits,
                    server_logits=None,  # No server predictions during evaluation
                    labels=labels,
                    attention_mask=attention_mask,
                    medical_mask=medical_mask
                )
                
                # Compute local predictions for metrics
                local_predictions = torch.softmax(local_logits, dim=-1)
                
                all_predictions.append(local_predictions)
                all_labels.append(labels)
                all_losses.append(loss_dict['task_loss'])  # Use task loss for evaluation
        
        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Move to CPU to save memory
        all_predictions = all_predictions.cpu()
        all_labels = all_labels.cpu()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        embed_dim = self.config.get('model', {}).get('embed_dim', 4096)
        sample_data = torch.randn(1, 10, embed_dim, device=self.device)
        
        # Prepare transmission
        quantized_data, scaling_params = self.privacy_mechanism.prepare_transmission(sample_data)
        reconstructed_data = self.privacy_mechanism.reconstruct_transmission(quantized_data, scaling_params)
        
        # Analyze privacy-utility tradeoff
        delta = self.config.get('privacy', {}).get('differential_privacy', {}).get('delta', 1e-5)
        analysis = self.privacy_mechanism.analyze_privacy_utility_tradeoff(
            sample_data, reconstructed_data, delta=delta
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
            'local_epochs': 2,
            'gradient_clip_norm': 1.0,
            'weight_decay': 0.01
        },
        'loss': {
            'temperature': 3.0,
            'distillation_weight': 0.5,
            'task_weight': 0.5,
            'medical_weight': 0.1,
            'label_smoothing': 0.1,
            'medical_term_boost': 2.0
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
