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
from ..training.losses import FederatedKnowledgeDistillationLoss
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
        self.privacy_mechanism = PrivacyPreservingTransmission(
            sigma=config['privacy']['gaussian_noise']['sigma'],
            num_bits=config['privacy']['quantization']['bits'],
            device=device
        )
        
        # Privacy accountant
        self.privacy_accountant = DifferentialPrivacyAccountant(
            total_epsilon=config['privacy']['differential_privacy']['epsilon'],
            delta=config['privacy']['differential_privacy']['delta']
        )
        
        # Loss function
        self.loss_fn = FederatedKnowledgeDistillationLoss(
            temperature=config['training']['temperature'],
            alpha=config['loss_weights']['server']['lambda1'],
            beta=config['loss_weights']['server']['lambda2']
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
        
        if optimizer_config['name'] == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config['server_lr'],
                weight_decay=self.config['training']['weight_decay'],
                eps=optimizer_config['eps']
            )
        elif optimizer_config['name'] == 'SGD':
            return optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['server_lr'],
                momentum=optimizer_config['momentum'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")
    
    def register_client(self, client_id: int):
        """Register a client with the server"""
        self.connected_clients.add(client_id)
        self.logger.info(f"Registered client {client_id}. Total clients: {len(self.connected_clients)}")
    
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
            server_predictions = torch.softmax(server_logits, dim=-1)
            
            # Prepare client predictions
            client_predictions = torch.softmax(client_logits.to(self.device), dim=-1)
            
            # Compute server loss
            loss_dict = self.loss_fn(
                client_logits=server_logits,  # Server logits as "client" for server loss
                server_logits=client_logits.to(self.device),  # Client logits as "server" for distillation
                labels=cached_data['labels'],
                client_predictions=server_predictions,
                server_predictions=client_predictions
            )
            
            client_loss = loss_dict['total_loss']
            total_loss += client_loss
            batch_count += 1
            
            # Store for metrics computation
            all_server_predictions.append(server_predictions)
            all_labels.append(cached_data['labels'])
            
            # Store loss components
            for key, value in loss_dict.items():
                loss_components[key].append(value)
        
        if batch_count == 0:
            self.logger.warning("No valid client data for global training step")
            return {'total_loss': 0.0}
        
        # Average loss across clients
        avg_loss = total_loss / batch_count
        
        # Backward pass
        avg_loss.backward()
        
        # Gradient clipping
        if self.config['training']['gradient_clip_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip_norm']
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
        if self.current_round % self.config['training']['fedavg_frequency'] == 0:
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
        sample_data = torch.randn(1, 10, self.config['model']['embed_dim'], device=self.device)
        privacy_analysis = self.privacy_mechanism.analyze_privacy_utility_tradeoff(
            sample_data, sample_data,  # Using same data for simplicity
            delta=self.config['privacy']['differential_privacy']['delta']
        )
        
        round_epsilon = privacy_analysis['epsilon'] / self.config['training']['num_rounds']
        
        # Add to privacy accountant
        budget_ok = self.privacy_accountant.add_round_epsilon(round_epsilon)
        
        privacy_stats = self.privacy_accountant.get_privacy_analysis()
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
                loss = nn.CrossEntropyLoss()(
                    server_logits.view(-1, server_logits.size(-1)),
                    labels.view(-1)
                )
                
                all_predictions.append(server_predictions)
                all_labels.append(labels)
                all_losses.append(loss)
