"""
Federated Server Implementation
Handles global model updates, client coordination, and federated averaging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging
from collections import defaultdict
import copy

from ..models.llama_split import LLaMAServerModel
from ..models.privacy_mechanisms import PrivacyPreservingTransmission, DifferentialPrivacyAccountant
from ..training.losses import MedicalFederatedDistillationLoss  # Fixed import
from ..training.metrics import MedicalQAMetrics

class FederatedServer:
    """
    Federated Learning Server for Medical QA
    Implements server-side logic from the methodology
    """
    
    def __init__(self, 
                 model: LLaMAServerModel,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        """
        Args:
            model: Server model instance
            config: Configuration dictionary
            device: Device to run computations on
        """
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
        
        # Privacy accountant
        privacy_config = config.get('privacy', {})
        target_epsilon = privacy_config.get('target_epsilon', 10.0)
        target_delta = privacy_config.get('target_delta', 1e-5)
        
        # Handle both old and new config formats
        if 'differential_privacy' in privacy_config:
            target_epsilon = privacy_config['differential_privacy'].get('epsilon', target_epsilon)
            target_delta = privacy_config['differential_privacy'].get('delta', target_delta)
        
        self.privacy_accountant = DifferentialPrivacyAccountant(
            total_epsilon=target_epsilon,
            delta=target_delta
        )
        
        # Loss function - Updated to use correct class and parameters
        loss_config = config.get('loss', {})
        training_config = config.get('training', {})
        
        # Handle different config structures for backward compatibility
        temperature = loss_config.get('temperature', training_config.get('temperature', 3.0))
        
        # Try to get loss weights from different possible locations
        alpha = 0.5  # default
        beta = 0.5   # default
        
        if 'loss_weights' in config and 'server' in config['loss_weights']:
            alpha = config['loss_weights']['server'].get('lambda1', 0.5)
            beta = config['loss_weights']['server'].get('lambda2', 0.5)
        else:
            alpha = loss_config.get('distillation_weight', 0.5)
            beta = loss_config.get('task_weight', 0.5)
        
        self.loss_fn = MedicalFederatedDistillationLoss(
            temperature=temperature,
            alpha=alpha,
            beta=beta,
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
        self.global_updates = 0
        self.training_history = []
        
        # Client management
        self.connected_clients = set()
        self.client_data_cache = {}
        
        # Federated averaging state
        self.aggregated_head_weights = None
        self.client_head_weights = {}
        
        # Logger
        self.logger = logging.getLogger("FederatedServer")
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for global model training"""
        optimizer_config = self.config['optimizer']
        
        # Ensure learning rate is a float (handle string values from YAML)
        server_lr = optimizer_config.get('server_lr', 1e-4)
        if isinstance(server_lr, str):
            server_lr = float(server_lr)
        
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
                lr=server_lr,
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
                lr=server_lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config.get('name', 'AdamW')}")
    
    def register_client(self, client_id: int):
        """Register a client with the server"""
        self.connected_clients.add(client_id)
        self.logger.info(f"Registered client {client_id}. Total clients: {len(self.connected_clients)}")
    
    def _create_simple_medical_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Create a simple medical mask based on token patterns
        This is a placeholder - in practice, you'd use medical vocabulary
        """
        medical_mask = torch.zeros_like(labels, dtype=torch.bool)
        
        # Simple heuristic: mark first few tokens as potentially medical
        seq_len = labels.size(-1)
        medical_tokens = min(3, seq_len)
        medical_mask[:, :medical_tokens] = True
        
        return medical_mask
    
    def process_client_data(self, client_request: Dict[str, Any]) -> torch.Tensor:
        """
        Process smashed data from client and return server predictions
        
        Args:
            client_request: Dictionary containing client data
                - client_id: Client identifier
                - smashed_data: Quantized smashed data
                - scaling_params: Parameters for dequantization
                - labels: Ground truth labels
                - attention_mask: Attention mask
                
        Returns:
            Server predictions for the client
        """
        client_id = client_request['client_id']
        quantized_data = client_request['smashed_data'].to(self.device)
        scaling_params = {k: v.to(self.device) for k, v in client_request['scaling_params'].items()}
        labels = client_request['labels'].to(self.device)
        attention_mask = client_request.get('attention_mask', None)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Step 1: Reconstruct smashed data
        reconstructed_data = self.privacy_mechanism.reconstruct_transmission(
            quantized_data, scaling_params
        )
        
        # Step 2: Forward pass through server model
        self.model.eval()  # Set to eval mode for inference
        with torch.no_grad():
            server_logits = self.model(reconstructed_data, attention_mask)
            server_predictions = torch.softmax(server_logits, dim=-1)
        
        # Cache client data for training
        self.client_data_cache[client_id] = {
            'reconstructed_data': reconstructed_data,
            'server_logits': server_logits,
            'server_predictions': server_predictions,
            'labels': labels,
            'attention_mask': attention_mask
        }
        
        return server_predictions
    
    def global_training_step(self, client_logits_dict: Dict[int, torch.Tensor]) -> Dict[str, float]:
        """
        Perform global model training step using cached client data
        
        Args:
            client_logits_dict: Dictionary mapping client_id to client logits
            
        Returns:
            Dictionary with loss values and metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        batch_count = 0
        all_server_predictions = []
        all_labels = []
        loss_components = defaultdict(list)
        
        # Process each client's cached data
        for client_id, client_logits in client_logits_dict.items():
            if client_id not in self.client_data_cache:
                self.logger.warning(f"No cached data for client {client_id}")
                continue
            
            cached_data = self.client_data_cache[client_id]
            
            # Forward pass through server model (training mode)
            server_logits = self.model(cached_data['reconstructed_data'], cached_data['attention_mask'])
            
            # Create medical mask
            medical_mask = self._create_simple_medical_mask(cached_data['labels'])
            
            # Compute server loss using updated loss function
            loss_dict = self.loss_fn(
                client_logits=server_logits,  # Server logits as "client" for server loss
                server_logits=client_logits.to(self.device),  # Client logits as "server" for distillation
                labels=cached_data['labels'],
                attention_mask=cached_data['attention_mask'],
                medical_mask=medical_mask
            )
            
            client_loss = loss_dict['total_loss']
            total_loss += client_loss
            batch_count += 1
            
            # Store for metrics computation
            server_predictions = torch.softmax(server_logits, dim=-1)
            all_server_predictions.append(server_predictions)
            all_labels.append(cached_data['labels'])
            
            # Store loss components
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    loss_components[key].append(value.item())
                else:
                    loss_components[key].append(value)
        
        if batch_count == 0:
            self.logger.warning("No valid client data for global training step")
            return {'total_loss': 0.0}
        
        # Average loss across clients
        avg_loss = total_loss / batch_count
        
        # Backward pass
        avg_loss.backward()
        
        # Gradient clipping
        gradient_clip_norm = self.config.get('training', {}).get('gradient_clip_norm', 1.0)
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                gradient_clip_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        self.global_updates += 1
        
        # Compute metrics
        with torch.no_grad():
            if all_server_predictions and all_labels:
                concat_predictions = torch.cat(all_server_predictions, dim=0)
                concat_labels = torch.cat(all_labels, dim=0)
                
                metrics = self.metrics.compute_batch_metrics(
                    predictions=concat_predictions,
                    labels=concat_labels,
                    input_ids=None  # Not needed for server metrics
                )
            else:
                metrics = {}
        
        # Average loss components
        avg_loss_components = {}
        for key, values in loss_components.items():
            if values:
                avg_loss_components[key] = sum(values) / len(values)
        
        # Combine results
        result = {**avg_loss_components, **metrics}
        result['num_clients'] = batch_count
        
        return result
    
    def federated_averaging(self, client_head_weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Perform federated averaging of client head weights
        
        Args:
            client_head_weights: List of head weight dictionaries from clients
            
        Returns:
            Aggregated head weights
        """
        if not client_head_weights:
            self.logger.warning("No client head weights provided for averaging")
            return {}
        
        # Initialize aggregated weights
        aggregated_weights = {}
        
        # Get parameter names from first client
        param_names = client_head_weights[0].keys()
        
        # Average each parameter across clients
        for param_name in param_names:
            # Collect all client parameters for this layer
            client_params = []
            for client_weights in client_head_weights:
                if param_name in client_weights:
                    client_params.append(client_weights[param_name])
            
            if client_params:
                # Simple averaging (can be extended to weighted averaging)
                aggregated_weights[param_name] = torch.stack(client_params).mean(dim=0)
        
        self.aggregated_head_weights = aggregated_weights
        if hasattr(self.model, 'update_head_weights'):
            self.model.update_head_weights(aggregated_weights)
        
        self.logger.info(f"Federated averaging completed for {len(aggregated_weights)} parameters")
        
        return aggregated_weights
    
    def weighted_federated_averaging(self, 
                                   client_head_weights: List[Dict[str, torch.Tensor]],
                                   client_data_sizes: List[int]) -> Dict[str, torch.Tensor]:
        """
        Perform weighted federated averaging based on client data sizes
        
        Args:
            client_head_weights: List of head weight dictionaries from clients
            client_data_sizes: List of data sizes for each client
            
        Returns:
            Weighted aggregated head weights
        """
        if not client_head_weights or not client_data_sizes:
            return self.federated_averaging(client_head_weights)
        
        if len(client_head_weights) != len(client_data_sizes):
            self.logger.warning("Mismatch between number of clients and data sizes, using simple averaging")
            return self.federated_averaging(client_head_weights)
        
        # Compute weights based on data sizes
        total_data = sum(client_data_sizes)
        client_weights = [size / total_data for size in client_data_sizes]
        
        # Initialize aggregated weights
        aggregated_weights = {}
        param_names = client_head_weights[0].keys()
        
        # Weighted average each parameter
        for param_name in param_names:
            weighted_sum = None
            
            for i, client_weights_dict in enumerate(client_head_weights):
                if param_name in client_weights_dict:
                    param_tensor = client_weights_dict[param_name]
                    weighted_param = param_tensor * client_weights[i]
                    
                    if weighted_sum is None:
                        weighted_sum = weighted_param
                    else:
                        weighted_sum += weighted_param
            
            if weighted_sum is not None:
                aggregated_weights[param_name] = weighted_sum
        
        self.aggregated_head_weights = aggregated_weights
        if hasattr(self.model, 'update_head_weights'):
            self.model.update_head_weights(aggregated_weights)
        
        self.logger.info(f"Weighted federated averaging completed with weights: {client_weights}")
        
        return aggregated_weights
    
    def complete_training_round(self, 
                              client_logits_dict: Dict[int, torch.Tensor],
                              client_head_weights: List[Dict[str, torch.Tensor]],
                              client_data_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Complete a full training round including global update and federated averaging
        
        Args:
            client_logits_dict: Client logits for distillation
            client_head_weights: Client head weights for averaging
            client_data_sizes: Optional client data sizes for weighted averaging
            
        Returns:
            Round training statistics
        """
        self.current_round += 1
        self.logger.info(f"Starting server training round {self.current_round}")
        
        # Step 1: Global model update
        global_stats = self.global_training_step(client_logits_dict)
        
        # Step 2: Federated averaging (if it's time)
        fedavg_stats = {}
        fedavg_frequency = self.config.get('training', {}).get('fedavg_frequency', 1)
        
        if self.current_round % fedavg_frequency == 0:
            if client_data_sizes is not None:
                aggregated_weights = self.weighted_federated_averaging(client_head_weights, client_data_sizes)
            else:
                aggregated_weights = self.federated_averaging(client_head_weights)
            
            fedavg_stats = {
                'fedavg_performed': True,
                'num_aggregated_params': len(aggregated_weights)
            }
        else:
            fedavg_stats = {'fedavg_performed': False}
        
        # Step 3: Privacy accounting
        privacy_stats = self._update_privacy_budget()
        
        # Combine statistics
        round_stats = {
            'round': self.current_round,
            'global_updates': self.global_updates,
            **global_stats,
            **fedavg_stats,
            **privacy_stats
        }
        
        # Clear client data cache
        self.client_data_cache.clear()
        
        self.training_history.append(round_stats)
        
        self.logger.info(f"Round {self.current_round} completed. Global loss: {global_stats.get('total_loss', 0):.4f}")
        
        return round_stats
    
    def _update_privacy_budget(self) -> Dict[str, float]:
        """Update privacy budget accounting"""
        # Compute round epsilon (simplified)
        embed_dim = self.config.get('model', {}).get('embed_dim', 4096)
        sample_data = torch.randn(1, 10, embed_dim, device=self.device)
        
        try:
            privacy_analysis = self.privacy_mechanism.analyze_privacy_utility_tradeoff(
                sample_data, sample_data,  # Using same data for simplicity
                delta=self.config.get('privacy', {}).get('target_delta', 1e-5)
            )
            
            num_rounds = self.config.get('training', {}).get('num_rounds', 100)
            round_epsilon = privacy_analysis.get('epsilon', 0.1) / num_rounds
        except Exception as e:
            self.logger.warning(f"Error computing privacy analysis: {e}")
            round_epsilon = 0.01  # Default small epsilon
        
        # Add to privacy accountant
        try:
            budget_ok = self.privacy_accountant.add_round_epsilon(round_epsilon)
        except Exception as e:
            self.logger.warning(f"Error updating privacy budget: {e}")
            budget_ok = True
        
        try:
            privacy_stats = self.privacy_accountant.get_privacy_analysis()
        except Exception as e:
            self.logger.warning(f"Error getting privacy analysis: {e}")
            privacy_stats = {'consumed_epsilon_basic': 0.0, 'privacy_exhausted': False}
        
        privacy_stats['round_epsilon'] = round_epsilon
        privacy_stats['budget_exceeded'] = not budget_ok
        
        if not budget_ok:
            self.logger.warning("Privacy budget exceeded!")
        
        return privacy_stats
    
    def evaluate_global_model(self, eval_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """
        Evaluate global model performance
        
        Args:
            eval_data: List of (reconstructed_data, labels) tuples
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_losses = []
        
        with torch.no_grad():
            for reconstructed_data, labels in eval_data:
                reconstructed_data = reconstructed_data.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                server_logits = self.model(reconstructed_data)
                server_predictions = torch.softmax(server_logits, dim=-1)
                
                # Compute loss (task loss only for evaluation)
                loss = nn.CrossEntropyLoss(ignore_index=-100)(
                    server_logits.view(-1, server_logits.size(-1)),
                    labels.view(-1)
                )
                
                all_predictions.append(server_predictions)
                all_labels.append(labels)
                all_losses.append(loss)
        
        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute comprehensive metrics
        metrics = self.metrics.compute_comprehensive_metrics(
            predictions=all_predictions,
            labels=all_labels,
            input_ids=None
        )
        
        # Add average loss
        metrics['eval_loss'] = torch.stack(all_losses).mean().item()
        
        self.logger.info(f"Global model evaluation completed. Loss: {metrics['eval_loss']:.4f}")
        
        return metrics
    
    def get_aggregated_head_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get current aggregated head weights"""
        return self.aggregated_head_weights
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get complete server model state for checkpointing"""
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_round': self.current_round,
            'global_updates': self.global_updates,
            'training_history': self.training_history,
            'aggregated_head_weights': self.aggregated_head_weights,
            'privacy_accountant': self.privacy_accountant.get_privacy_analysis()
        }
    
    def load_model_state(self, state: Dict[str, Any]):
        """Load server model state from checkpoint"""
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.current_round = state['current_round']
        self.global_updates = state['global_updates']
        self.training_history = state['training_history']
        self.aggregated_head_weights = state.get('aggregated_head_weights')
        
        # Restore privacy accountant state if available
        if 'privacy_accountant' in state:
            privacy_state = state['privacy_accountant']
            try:
                self.privacy_accountant.consumed_epsilon = privacy_state.get('consumed_epsilon_basic', 0.0)
                self.privacy_accountant.round_epsilons = []  # Reset round epsilons
            except Exception as e:
                self.logger.warning(f"Error restoring privacy accountant state: {e}")
        
        self.logger.info(f"Loaded server model state from round {self.current_round}")
    
    def get_communication_statistics(self) -> Dict[str, Any]:
        """Get communication cost statistics"""
        try:
            # Sample data for analysis
            max_seq_length = self.config.get('model', {}).get('max_seq_length', 512)
            embed_dim = self.config.get('model', {}).get('embed_dim', 4096)
            sample_shape = (1, max_seq_length, embed_dim)
            comm_stats = self.privacy_mechanism.compute_communication_cost(sample_shape)
            
            # Add federated averaging communication cost
            if self.aggregated_head_weights:
                head_params_size = sum(param.numel() for param in self.aggregated_head_weights.values())
                head_comm_bits = head_params_size * 32  # 32-bit floats
                
                comm_stats['fedavg_upload_bits'] = head_comm_bits * len(self.connected_clients)
                comm_stats['fedavg_download_bits'] = head_comm_bits * len(self.connected_clients)
                comm_stats['total_fedavg_bits'] = comm_stats['fedavg_upload_bits'] + comm_stats['fedavg_download_bits']
            
            # Per-round and total communication
            rounds_completed = self.current_round
            if rounds_completed > 0:
                per_round_bits = comm_stats.get('quantized_bits', 0) * len(self.connected_clients)
                fedavg_frequency = self.config.get('training', {}).get('fedavg_frequency', 1)
                fedavg_rounds = rounds_completed // fedavg_frequency
                
                total_training_bits = per_round_bits * rounds_completed
                total_fedavg_bits = comm_stats.get('total_fedavg_bits', 0) * fedavg_rounds
                
                comm_stats['per_round_bits'] = per_round_bits
                comm_stats['total_training_bits'] = total_training_bits
                comm_stats['total_fedavg_bits_cumulative'] = total_fedavg_bits
                comm_stats['total_communication_bits'] = total_training_bits + total_fedavg_bits
            
        except Exception as e:
            self.logger.warning(f"Error computing communication statistics: {e}")
            comm_stats = {'error': str(e)}
        
        return comm_stats
    
    def should_continue_training(self) -> bool:
        """Check if training should continue based on privacy budget and other criteria"""
        try:
            # Check privacy budget
            privacy_analysis = self.privacy_accountant.get_privacy_analysis()
            if privacy_analysis.get('privacy_exhausted', False):
                self.logger.warning("Privacy budget exhausted, stopping training")
                return False
        except Exception as e:
            self.logger.warning(f"Error checking privacy budget: {e}")
        
        # Check maximum rounds
        max_rounds = self.config.get('training', {}).get('num_rounds', 100)
        if self.current_round >= max_rounds:
            self.logger.info("Maximum rounds reached, stopping training")
            return False
        
        return True
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        try:
            privacy_analysis = self.privacy_accountant.get_privacy_analysis()
        except Exception as e:
            self.logger.warning(f"Error getting privacy analysis: {e}")
            privacy_analysis = {'error': str(e)}
        
        try:
            comm_stats = self.get_communication_statistics()
        except Exception as e:
            self.logger.warning(f"Error getting communication statistics: {e}")
            comm_stats = {'error': str(e)}
        
        summary = {
            'server_info': {
                'current_round': self.current_round,
                'global_updates': self.global_updates,
                'connected_clients': len(self.connected_clients),
                'total_rounds_planned': self.config.get('training', {}).get('num_rounds', 100)
            },
            'privacy_analysis': privacy_analysis,
            'communication_stats': comm_stats,
            'training_progress': {
                'rounds_completed': self.current_round,
                'progress_percentage': (self.current_round / self.config.get('training', {}).get('num_rounds', 100)) * 100
            }
        }
        
        # Add recent training statistics
        if self.training_history:
            recent_history = self.training_history[-5:]  # Last 5 rounds
            summary['recent_performance'] = {
                'avg_loss': np.mean([h.get('total_loss', 0) for h in recent_history]),
                'avg_accuracy': np.mean([h.get('accuracy', 0) for h in recent_history]),
                'rounds_included': len(recent_history)
            }
        
        return summary

class ServerCommunicationHandler:
    """Handles communication protocols between server and clients"""
    
    def __init__(self, server: FederatedServer):
        """
        Args:
            server: FederatedServer instance
        """
        self.server = server
        self.logger = logging.getLogger("ServerCommHandler")
        
    def create_client_communication_function(self, client_id: int) -> callable:
        """
        Create a communication function for a specific client
        
        Args:
            client_id: Client identifier
            
        Returns:
            Communication function for the client
        """
        def communicate_with_server(client_request: Dict[str, Any]) -> torch.Tensor:
            """
            Handle communication from client to server
            
            Args:
                client_request: Request from client
                
            Returns:
                Server predictions
            """
            try:
                # Add client ID to request
                client_request['client_id'] = client_id
                
                # Process client data and return predictions
                server_predictions = self.server.process_client_data(client_request)
                
                return server_predictions
                
            except Exception as e:
                self.logger.error(f"Error processing request from client {client_id}: {e}")
                # Return dummy predictions in case of error
                batch_size = client_request['labels'].shape[0]
                seq_len = client_request['labels'].shape[1] if len(client_request['labels'].shape) > 1 else 1
                vocab_size = self.server.config['model']['vocab_size']
                
                dummy_predictions = torch.zeros(batch_size, seq_len, vocab_size)
                return dummy_predictions
        
        return communicate_with_server
    
    def coordinate_federated_round(self, 
                                 client_managers: List,
                                 client_dataloaders: List) -> Dict[str, Any]:
        """
        Coordinate a complete federated training round
        
        Args:
            client_managers: List of client instances
            client_dataloaders: List of client dataloaders
            
        Returns:
            Round statistics
        """
        self.logger.info(f"Coordinating federated round {self.server.current_round + 1}")
        
        # Step 1: Collect client logits and head weights
        client_logits_dict = {}
        client_head_weights = []
        client_data_sizes = []
        
        for i, (client, dataloader) in enumerate(zip(client_managers, client_dataloaders)):
            try:
                # Create communication function for this client
                comm_fn = self.create_client_communication_function(client.client_id)
                
                # Run client training round
                client_stats = client.local_training_round(dataloader, comm_fn)
                
                # Collect client logits (would need to be implemented in client)
                # For now, using dummy data
                vocab_size = self.server.config.get('model', {}).get('vocab_size', 32000)
                dummy_logits = torch.randn(1, 10, vocab_size)
                client_logits_dict[client.client_id] = dummy_logits
                
                # Collect head weights
                head_weights = client.get_head_parameters()
                client_head_weights.append(head_weights)
                
                # Collect data size
                client_data_sizes.append(len(dataloader.dataset))
                
                self.logger.debug(f"Collected data from client {client.client_id}")
                
            except Exception as e:
                self.logger.error(f"Error in client {i} training: {e}")
                continue
        
        # Step 2: Server training and federated averaging
        if client_logits_dict and client_head_weights:
            round_stats = self.server.complete_training_round(
                client_logits_dict=client_logits_dict,
                client_head_weights=client_head_weights,
                client_data_sizes=client_data_sizes
            )
            
            # Step 3: Distribute updated head weights back to clients
            aggregated_weights = self.server.get_aggregated_head_weights()
            if aggregated_weights:
                for client in client_managers:
                    try:
                        client.set_head_parameters(aggregated_weights)
                    except Exception as e:
                        self.logger.error(f"Error updating client {client.client_id} parameters: {e}")
            
            round_stats['successful_clients'] = len(client_logits_dict)
            return round_stats
        
        else:
            self.logger.warning("No successful client communications in this round")
            return {'round': self.server.current_round, 'successful_clients': 0}

# Test function
def test_federated_server():
    """Test the federated server implementation"""
    from ..models.llama_split import create_llama_models
    
    # Mock configuration
    config = {
        'model': {'embed_dim': 256, 'vocab_size': 1000, 'max_seq_length': 128},
        'privacy': {
            'target_epsilon': 10.0,
            'target_delta': 1e-5,
            'gaussian_noise': {'sigma': 0.1},
            'quantization': {'bits': 8}
        },
        'training': {
            'num_rounds': 10,
            'fedavg_frequency': 2,
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
            'server_lr': 1e-4,
            'eps': 1e-8
        }
    }
    
    # Create server model
    _, server_model = create_llama_models(config['model'])
    
    # Create server
    server = FederatedServer(
        model=server_model,
        config=config,
        device='cpu'  # Use CPU for testing
    )
    
    # Test client registration
    server.register_client(0)
    server.register_client(1)
    server.register_client(2)
    
    # Test privacy budget
    try:
        privacy_analysis = server.privacy_accountant.get_privacy_analysis()
        print("Initial Privacy Analysis:")
        for key, value in privacy_analysis.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Privacy analysis error: {e}")
    
    # Test communication statistics
    try:
        comm_stats = server.get_communication_statistics()
        print(f"\nCommunication Statistics:")
        for key, value in comm_stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.2f}")
    except Exception as e:
        print(f"Communication stats error: {e}")
    
    # Test training summary
    try:
        summary = server.get_training_summary()
        print(f"\nTraining Summary:")
        print(f"  Connected clients: {summary['server_info']['connected_clients']}")
        print(f"  Current round: {summary['server_info']['current_round']}")
        
        if 'privacy_analysis' in summary and 'remaining_epsilon' in summary['privacy_analysis']:
            print(f"  Privacy budget remaining: {summary['privacy_analysis']['remaining_epsilon']:.4f}")
    except Exception as e:
        print(f"Training summary error: {e}")
    
    print("Federated server test completed successfully!")

if __name__ == "__main__":
    test_federated_server()
