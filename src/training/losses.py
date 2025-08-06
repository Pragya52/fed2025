"""
Enhanced Loss Functions for Medical Federated Learning
Includes medical-specific loss components and federated distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import numpy as np
import math

class MedicalFederatedDistillationLoss(nn.Module):
    """
    Enhanced federated knowledge distillation loss with medical-specific components
    """
    
    def __init__(self,
                 temperature: float = 3.0,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 gamma: float = 0.1,  # Medical loss weight
                 label_smoothing: float = 0.1,
                 medical_term_boost: float = 2.0):
        """
        Args:
            temperature: Knowledge distillation temperature
            alpha: Weight for distillation loss
            beta: Weight for task loss  
            gamma: Weight for medical-specific loss
            label_smoothing: Label smoothing factor
            medical_term_boost: Boost factor for medical terms
        """
        super().__init__()
        
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.medical_term_boost = medical_term_boost
        
        # Loss functions
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=-100
        )
        
        # Medical vocabulary for term identification
        self.medical_vocab_ids = self._create_medical_vocab_mapping()
        
    def _create_medical_vocab_mapping(self) -> set:
        """Create mapping of medical vocabulary IDs"""
        # This would be populated with actual medical term token IDs
        # For now, we'll use a placeholder approach
        return set(range(1000, 2000))  # Placeholder medical token IDs
    
    def forward(self,
                client_logits: torch.Tensor,
                server_logits: Optional[torch.Tensor],
                labels: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                medical_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute enhanced federated distillation loss
        
        Args:
            client_logits: Client model logits [batch, seq, vocab]
            server_logits: Server model logits [batch, seq, vocab] 
            labels: Ground truth labels [batch, seq]
            attention_mask: Attention mask [batch, seq]
            medical_mask: Medical term mask [batch, seq]
            
        Returns:
            Dictionary of loss components
        """
        batch_size, seq_len, vocab_size = client_logits.shape
        
        # Flatten for loss computation
        client_logits_flat = client_logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # Task loss (cross-entropy with ground truth)  
        task_loss = self.cross_entropy_loss(client_logits_flat, labels_flat)
        
        # Initialize loss dictionary
        loss_dict = {
            'task_loss': task_loss,
            'distillation_loss': torch.tensor(0.0, device=client_logits.device),
            'medical_loss': torch.tensor(0.0, device=client_logits.device),
            'consistency_loss': torch.tensor(0.0, device=client_logits.device),
            'total_loss': task_loss
        }
        
        # Knowledge distillation loss
        if server_logits is not None:
            distillation_loss = self._compute_distillation_loss(
                client_logits, server_logits, attention_mask
            )
            loss_dict['distillation_loss'] = distillation_loss
            
            # Consistency loss between client and server
            consistency_loss = self._compute_consistency_loss(
                client_logits, server_logits, attention_mask
            )
            loss_dict['consistency_loss'] = consistency_loss
        
        # Medical-specific loss
        if medical_mask is not None or self.gamma > 0:
            medical_loss = self._compute_medical_loss(
                client_logits, labels, medical_mask, attention_mask
            )
            loss_dict['medical_loss'] = medical_loss
        
        # Combine losses
        total_loss = (
            self.beta * task_loss +
            self.alpha * loss_dict['distillation_loss'] +
            self.gamma * loss_dict['medical_loss'] +
            0.1 * loss_dict['consistency_loss']  # Small weight for consistency
        )
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict
    
    def _compute_distillation_loss(self,
                                 client_logits: torch.Tensor,
                                 server_logits: torch.Tensor,
                                 attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute knowledge distillation loss"""
        
        # Temperature-scaled softmax
        client_soft = F.log_softmax(client_logits / self.temperature, dim=-1)
        server_soft = F.softmax(server_logits / self.temperature, dim=-1)
        
        # Reshape for KL divergence
        client_soft_flat = client_soft.view(-1, client_soft.size(-1))
        server_soft_flat = server_soft.view(-1, server_soft.size(-1))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.view(-1).bool()
            if mask.sum() > 0:
                client_soft_flat = client_soft_flat[mask]
                server_soft_flat = server_soft_flat[mask]
        
        # KL divergence loss
        kl_loss = self.kl_div_loss(client_soft_flat, server_soft_flat)
        
        # Scale by temperature squared
        kl_loss *= (self.temperature ** 2)
        
        return kl_loss
    
    def _compute_consistency_loss(self,
                                client_logits: torch.Tensor,
                                server_logits: torch.Tensor,
                                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute consistency loss between client and server predictions"""
        
        client_probs = F.softmax(client_logits, dim=-1)
        server_probs = F.softmax(server_logits, dim=-1)
        
        # MSE loss between probability distributions
        mse_loss = F.mse_loss(client_probs, server_probs, reduction='none').mean(dim=-1)
        
        # Apply attention mask
        if attention_mask is not None:
            mse_loss = mse_loss * attention_mask
            return mse_loss.sum() / attention_mask.sum().clamp(min=1)
        else:
            return mse_loss.mean()
    
    def _compute_medical_loss(self,
                            logits: torch.Tensor,
                            labels: torch.Tensor,
                            medical_mask: Optional[torch.Tensor] = None,
                            attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute medical-specific loss with term boosting"""
        
        batch_size, seq_len, vocab_size = logits.shape
        
        if medical_mask is None:
            # Create medical mask based on labels
            medical_mask = self._create_medical_mask_from_labels(labels)
        
        # Flatten tensors
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        medical_mask_flat = medical_mask.view(-1)
        
        # Create combined mask
        valid_mask = (labels_flat != -100)
        if attention_mask is not None:
            attention_mask_flat = attention_mask.view(-1)
            valid_mask = valid_mask & attention_mask_flat.bool()
        
        combined_mask = valid_mask & medical_mask_flat.bool()
        
        if combined_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Compute weighted cross-entropy for medical terms
        medical_logits = logits_flat[combined_mask]
        medical_labels = labels_flat[combined_mask]
        
        # Apply boosting to medical terms
        medical_loss = F.cross_entropy(
            medical_logits, medical_labels, reduction='mean'
        )
        
        # Apply medical term boost
        medical_loss = medical_loss * self.medical_term_boost
        
        return medical_loss
    
    def _create_medical_mask_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Create medical mask based on label tokens"""
        medical_mask = torch.zeros_like(labels, dtype=torch.bool)
        
        # Mark positions where labels contain medical vocabulary
        for vocab_id in self.medical_vocab_ids:
            medical_mask |= (labels == vocab_id)
        
        return medical_mask

class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss for handling class imbalance in medical QA
    Dynamically adjusts focusing parameter based on difficulty
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 gamma_init: float = 2.0,
                 adaptive: bool = True,
                 reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for rare classes
            gamma_init: Initial focusing parameter
            adaptive: Whether to adapt gamma during training
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma_init
        self.adaptive = adaptive
        self.reduction = reduction
        
        # Track accuracy for adaptive gamma
        self.running_accuracy = 0.9
        self.momentum = 0.99
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive focal loss
        
        Args:
            inputs: Predicted logits [batch_size, num_classes] or [batch_size, seq_len, num_classes]
            targets: Ground truth labels
            
        Returns:
            Focal loss
        """
        # Handle different input shapes
        if inputs.dim() == 3:  # [batch, seq, vocab]
            inputs_flat = inputs.view(-1, inputs.size(-1))
            targets_flat = targets.view(-1)
        else:
            inputs_flat = inputs
            targets_flat = targets
        
        # Filter out ignored tokens
        mask = targets_flat != -100
        if mask.sum() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        inputs_filtered = inputs_flat[mask]
        targets_filtered = targets_flat[mask]
        
        # Compute cross-entropy
        ce_loss = F.cross_entropy(inputs_filtered, targets_filtered, reduction='none')
        
        # Compute pt (probability of true class)
        pt = torch.exp(-ce_loss)
        
        # Update running accuracy for adaptive gamma
        if self.adaptive and self.training:
            current_acc = (inputs_filtered.argmax(dim=1) == targets_filtered).float().mean()
            self.running_accuracy = self.momentum * self.running_accuracy + (1 - self.momentum) * current_acc
            
            # Adapt gamma based on accuracy (higher gamma for lower accuracy)
            self.gamma = 2.0 + 2.0 * (1.0 - self.running_accuracy)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ContrastiveMedicalLoss(nn.Module):
    """
    Contrastive loss for medical QA to improve representation learning
    """
    
    def __init__(self, 
                 temperature: float = 0.07,
                 medical_boost: float = 1.5):
        """
        Args:
            temperature: Temperature parameter for contrastive loss
            medical_boost: Boost factor for medical term similarities
        """
        super().__init__()
        self.temperature = temperature
        self.medical_boost = medical_boost
        
    def forward(self,
                client_features: torch.Tensor,
                server_features: torch.Tensor,
                labels: torch.Tensor,
                medical_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute contrastive loss between client and server features
        
        Args:
            client_features: Client feature representations [batch_size, feature_dim]
            server_features: Server feature representations [batch_size, feature_dim]
            labels: Labels for determining positive/negative pairs
            medical_mask: Mask for medical terms
            
        Returns:
            Contrastive loss
        """
        batch_size = client_features.size(0)
        
        # Normalize features
        client_features = F.normalize(client_features, dim=1)
        server_features = F.normalize(server_features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(client_features, server_features.T) / self.temperature
        
        # Create positive pair mask (same labels)
        labels_expanded = labels.unsqueeze(0)
        positive_mask = torch.eq(labels_expanded, labels_expanded.T).float()
        
        # Apply medical boost to medical term pairs
        if medical_mask is not None:
            medical_expanded = medical_mask.unsqueeze(0)
            medical_pair_mask = torch.logical_and(medical_expanded, medical_expanded.T).float()
            similarity_matrix = similarity_matrix + medical_pair_mask * self.medical_boost
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        
        # Positive pairs
        positive_sim = exp_sim * positive_mask
        
        # All pairs (for normalization)
        all_sim = exp_sim.sum(dim=1, keepdim=True)
        
        # Avoid division by zero
        positive_sum = positive_sim.sum(dim=1).clamp(min=1e-8)
        all_sum = all_sim.squeeze(1).clamp(min=1e-8)
        
        # Contrastive loss
        loss = -torch.log(positive_sum / all_sum)
        
        return loss.mean()

class UncertaintyAwareLoss(nn.Module):
    """
    Uncertainty-aware loss that adapts based on model confidence
    """
    
    def __init__(self, 
                 base_loss: nn.Module,
                 uncertainty_weight: float = 0.1):
        """
        Args:
            base_loss: Base loss function
            uncertainty_weight: Weight for uncertainty regularization
        """
        super().__init__()
        self.base_loss = base_loss
        self.uncertainty_weight = uncertainty_weight
        
    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
        Compute uncertainty-aware loss
        
        Args:
            inputs: Model predictions
            targets: Ground truth
            
        Returns:
            Uncertainty-aware loss
        """
        # Compute base loss
        base_loss_val = self.base_loss(inputs, targets, **kwargs)
        
        # Compute prediction confidence (entropy-based uncertainty)
        if inputs.dim() == 3:
            probs = F.softmax(inputs, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            
            # Mask for valid tokens
            mask = targets != -100
            if mask.sum() > 0:
                avg_uncertainty = entropy[mask].mean()
            else:
                avg_uncertainty = torch.tensor(0.0, device=inputs.device)
        else:
            probs = F.softmax(inputs, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            avg_uncertainty = entropy.mean()
        
        # Uncertainty regularization (encourage confident predictions)
        uncertainty_loss = self.uncertainty_weight * avg_uncertainty
        
        return base_loss_val + uncertainty_loss

class CombinedMedicalLoss(nn.Module):
    """
    Combined loss function integrating multiple medical-specific components
    """
    
    def __init__(self, config: Dict):
        """
        Initialize combined medical loss with configuration
        
        Args:
            config: Loss configuration dictionary
        """
        super().__init__()
        
        # Base federated distillation loss
        self.distillation_loss = MedicalFederatedDistillationLoss(
            temperature=config.get('temperature', 3.0),
            alpha=config.get('distillation_weight', 0.5),
            beta=config.get('task_weight', 0.5),
            gamma=config.get('medical_weight', 0.1),
            label_smoothing=config.get('label_smoothing', 0.1),
            medical_term_boost=config.get('medical_term_boost', 2.0)
        )
        
        # Additional loss components
        self.use_focal = config.get('use_focal_loss', True)
        if self.use_focal:
            self.focal_loss = AdaptiveFocalLoss(
                alpha=config.get('focal_alpha', 1.0),
                gamma_init=config.get('focal_gamma', 2.0),
                adaptive=config.get('adaptive_focal', True)
            )
        
        self.use_contrastive = config.get('use_contrastive_loss', True)
        if self.use_contrastive:
            self.contrastive_loss = ContrastiveMedicalLoss(
                temperature=config.get('contrastive_temperature', 0.07),
                medical_boost=config.get('contrastive_medical_boost', 1.5)
            )
        
        self.use_uncertainty = config.get('use_uncertainty_loss', False)
        if self.use_uncertainty:
            self.uncertainty_loss = UncertaintyAwareLoss(
                base_loss=nn.CrossEntropyLoss(ignore_index=-100),
                uncertainty_weight=config.get('uncertainty_weight', 0.1)
            )
        
        # Loss weights
        self.focal_weight = config.get('focal_weight', 0.2)
        self.contrastive_weight = config.get('contrastive_weight', 0.1)
        self.uncertainty_weight = config.get('uncertainty_loss_weight', 0.1)
        
    def forward(self,
                client_logits: torch.Tensor,
                server_logits: Optional[torch.Tensor],
                labels: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                medical_mask: Optional[torch.Tensor] = None,
                client_features: Optional[torch.Tensor] = None,
                server_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined medical loss
        
        Returns:
            Dictionary with all loss components
        """
        # Base distillation loss
        loss_dict = self.distillation_loss(
            client_logits, server_logits, labels, attention_mask, medical_mask
        )
        
        total_loss = loss_dict['total_loss']
        
        # Add focal loss
        if self.use_focal:
            focal_loss_val = self.focal_loss(client_logits, labels)
            loss_dict['focal_loss'] = focal_loss_val
            total_loss += self.focal_weight * focal_loss_val
        
        # Add contrastive loss
        if self.use_contrastive and client_features is not None and server_features is not None:
            # Pool features for contrastive learning
            if client_features.dim() == 3:  # [batch, seq, dim]
                client_pooled = client_features.mean(dim=1)
                server_pooled = server_features.mean(dim=1)
            else:
                client_pooled = client_features
                server_pooled = server_features
            
            # Extract sequence-level labels for contrastive learning
            if labels.dim() == 2:  # [batch, seq]
                seq_labels = labels[:, 0]  # Use first token as sequence label
            else:
                seq_labels = labels
            
            contrastive_loss_val = self.contrastive_loss(
                client_pooled, server_pooled, seq_labels, 
                medical_mask.any(dim=1) if medical_mask is not None else None
            )
            loss_dict['contrastive_loss'] = contrastive_loss_val
            total_loss += self.contrastive_weight * contrastive_loss_val
        
        # Add uncertainty loss
        if self.use_uncertainty:
            uncertainty_loss_val = self.uncertainty_loss(client_logits, labels)
            loss_dict['uncertainty_loss'] = uncertainty_loss_val
            total_loss += self.uncertainty_weight * uncertainty_loss_val
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict

class PrivacyAwareLoss(nn.Module):
    """
    Privacy-aware loss that adds noise to gradients for differential privacy
    """
    
    def __init__(self, 
                 base_loss: nn.Module,
                 noise_multiplier: float = 0.1,
                 max_grad_norm: float = 1.0):
        """
        Args:
            base_loss: Base loss function
            noise_multiplier: Gaussian noise multiplier for DP
            max_grad_norm: Maximum gradient norm for clipping
        """
        super().__init__()
        self.base_loss = base_loss
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Compute base loss (privacy is handled during backward pass)"""
        return self.base_loss(*args, **kwargs)
    
    def add_privacy_noise_to_gradients(self, model: nn.Module):
        """Add privacy noise to model gradients"""
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        
        # Add Gaussian noise
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.noise_multiplier
                param.grad += noise

# Utility functions
def create_medical_loss_function(config: Dict) -> nn.Module:
    """
    Factory function to create medical loss function based on configuration
    
    Args:
        config: Loss configuration
        
    Returns:
        Medical loss function
    """
    loss_type = config.get('loss_type', 'combined')
    
    if loss_type == 'combined':
        return CombinedMedicalLoss(config)
    elif loss_type == 'distillation':
        return MedicalFederatedDistillationLoss(
            temperature=config.get('temperature', 3.0),
            alpha=config.get('distillation_weight', 0.5),
            beta=config.get('task_weight', 0.5),
            gamma=config.get('medical_weight', 0.1)
        )
    elif loss_type == 'focal':
        return AdaptiveFocalLoss(
            alpha=config.get('focal_alpha', 1.0),
            gamma_init=config.get('focal_gamma', 2.0)
        )
    else:
        # Default to cross-entropy
        return nn.CrossEntropyLoss(
            label_smoothing=config.get('label_smoothing', 0.0),
            ignore_index=-100
        )

def compute_loss_statistics(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute statistics from loss dictionary"""
    stats = {}
    
    for loss_name, loss_value in loss_dict.items():
        if isinstance(loss_value, torch.Tensor):
            stats[loss_name] = loss_value.item()
            
            # Add relative contribution to total loss
            if loss_name != 'total_loss' and 'total_loss' in loss_dict:
                total_loss = loss_dict['total_loss'].item()
                if total_loss > 1e-8:
                    stats[f'{loss_name}_ratio'] = loss_value.item() / total_loss
    
    return stats

# Test function
def test_enhanced_loss_functions():
    """Test enhanced loss functions"""
    print("Testing Enhanced Medical Loss Functions...")
    
    # Test data
    batch_size, seq_len, vocab_size = 4, 10, 1000
    
    client_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    server_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    medical_mask = torch.zeros(batch_size, seq_len).bool()
    medical_mask[:, :3] = True  # First 3 tokens are medical
    
    # Test individual loss functions
    print("1. Testing Medical Federated Distillation Loss...")
    distill_loss = MedicalFederatedDistillationLoss()
    loss_dict = distill_loss(client_logits, server_logits, labels, attention_mask, medical_mask)
    
    for key, value in loss_dict.items():
        print(f"   {key}: {value.item():.4f}")
    
    print("\n2. Testing Adaptive Focal Loss...")
    focal_loss = AdaptiveFocalLoss()
    focal_loss_val = focal_loss(client_logits, labels)
    print(f"   Focal loss: {focal_loss_val.item():.4f}")
    
    print("\n3. Testing Combined Medical Loss...")
    config = {
        'temperature': 3.0,
        'distillation_weight': 0.5,
        'task_weight': 0.5,
        'medical_weight': 0.1,
        'use_focal_loss': True,
        'use_contrastive_loss': True,
        'focal_weight': 0.2,
        'contrastive_weight': 0.1
    }
    
    combined_loss = CombinedMedicalLoss(config)
    
    # Create feature tensors
    client_features = torch.randn(batch_size, seq_len, 256)
    server_features = torch.randn(batch_size, seq_len, 256)
    
    combined_loss_dict = combined_loss(
        client_logits, server_logits, labels, attention_mask, medical_mask,
        client_features, server_features
    )
    
    print("   Combined loss components:")
    for key, value in combined_loss_dict.items():
        print(f"     {key}: {value.item():.4f}")
    
    # Test loss statistics
    print("\n4. Testing loss statistics...")
    stats = compute_loss_statistics(combined_loss_dict)
    for key, value in stats.items():
        print(f"   {key}: {value:.4f}")
    
    # Test backward pass
    print("\n5. Testing backward pass...")
    total_loss = combined_loss_dict['total_loss']
    total_loss.backward()
    
    print(f"   Client logits gradient norm: {client_logits.grad.norm().item():.4f}")
    print(f"   Server logits gradient norm: {server_logits.grad.norm().item():.4f}")
    
    print("\nEnhanced Loss Functions test completed!")

if __name__ == "__main__":
    test_enhanced_loss_functions()
