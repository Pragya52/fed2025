"""
Loss Functions for Federated Split Learning
Implements knowledge distillation and task-specific losses from the methodology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

class FederatedKnowledgeDistillationLoss(nn.Module):
    """
    Implements the bidirectional knowledge distillation loss from the methodology
    
    For client: L_client = μ₁ * L_CE(client_soft, server_soft) + μ₂ * L_CE(client_pred, y)
    For server: L_server = λ₁ * L_CE(server_soft, client_soft) + λ₂ * L_CE(server_pred, y)
    """
    
    def __init__(self, 
                 temperature: float = 3.0,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 label_smoothing: float = 0.0):
        """
        Args:
            temperature: Temperature parameter τ for knowledge distillation
            alpha: Weight for distillation loss (μ₁ or λ₁)
            beta: Weight for task loss (μ₂ or λ₂)
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.label_smoothing = label_smoothing
        
        # Loss functions
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def forward(self, 
                client_logits: torch.Tensor,
                server_logits: Optional[torch.Tensor],
                labels: torch.Tensor,
                client_predictions: Optional[torch.Tensor] = None,
                server_predictions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute federated knowledge distillation loss
        
        Args:
            client_logits: Logits from client model
            server_logits: Logits from server model (None for evaluation)
            labels: Ground truth labels
            client_predictions: Pre-computed client predictions (optional)
            server_predictions: Pre-computed server predictions (optional)
            
        Returns:
            Dictionary containing different loss components
        """
        batch_size, seq_len, vocab_size = client_logits.shape
        
        # Compute predictions if not provided
        if client_predictions is None:
            client_predictions = F.softmax(client_logits, dim=-1)
        
        if server_predictions is None and server_logits is not None:
            server_predictions = F.softmax(server_logits, dim=-1)
        
        # Task loss (cross-entropy with ground truth)
        task_loss = self.cross_entropy_loss(
            client_logits.view(-1, vocab_size),
            labels.view(-1)
        )
        
        # Initialize loss dictionary
        loss_dict = {
            'task_loss': task_loss,
            'distillation_loss': torch.tensor(0.0, device=client_logits.device),
            'total_loss': task_loss
        }
        
        # Knowledge distillation loss (if server predictions available)
        if server_logits is not None and server_predictions is not None:
            # Temperature-scaled softmax for distillation
            client_soft = F.softmax(client_logits / self.temperature, dim=-1)
            server_soft = F.softmax(server_logits / self.temperature, dim=-1)
            
            # KL divergence loss for knowledge distillation
            # Note: KL(server_soft || client_soft) for client learning from server
            distillation_loss = self.kl_div_loss(
                F.log_softmax(client_logits / self.temperature, dim=-1).view(-1, vocab_size),
                server_soft.view(-1, vocab_size)
            )
            
            # Scale by temperature squared (standard practice)
            distillation_loss *= (self.temperature ** 2)
            
            loss_dict['distillation_loss'] = distillation_loss
            
            # Combined loss
            total_loss = self.alpha * distillation_loss + self.beta * task_loss
            loss_dict['total_loss'] = total_loss
        
        else:
            # No distillation, only task loss
            loss_dict['total_loss'] = self.beta * task_loss
        
        return loss_dict

class MedicalQALoss(nn.Module):
    """
    Specialized loss function for medical QA tasks
    Includes additional components for medical domain
    """
    
    def __init__(self, 
                 base_loss: FederatedKnowledgeDistillationLoss,
                 medical_weight: float = 0.1,
                 consistency_weight: float = 0.1):
        """
        Args:
            base_loss: Base federated distillation loss
            medical_weight: Weight for medical-specific loss components
            consistency_weight: Weight for consistency regularization
        """
        super().__init__()
        self.base_loss = base_loss
        self.medical_weight = medical_weight
        self.consistency_weight = consistency_weight
        
    def forward(self, 
                client_logits: torch.Tensor,
                server_logits: Optional[torch.Tensor],
                labels: torch.Tensor,
                client_predictions: Optional[torch.Tensor] = None,
                server_predictions: Optional[torch.Tensor] = None,
                medical_masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute medical QA loss with domain-specific components
        
        Args:
            medical_masks: Binary mask for medical-specific tokens (optional)
            Other args same as base loss
            
        Returns:
            Loss dictionary with medical components
        """
        # Get base loss
        loss_dict = self.base_loss(
            client_logits, server_logits, labels, 
            client_predictions, server_predictions
        )
        
        # Medical-specific loss components
        medical_loss = torch.tensor(0.0, device=client_logits.device)
        consistency_loss = torch.tensor(0.0, device=client_logits.device)
        
        # Medical token focus (if medical masks provided)
        if medical_masks is not None:
            medical_logits = client_logits * medical_masks.unsqueeze(-1)
            medical_labels = labels * medical_masks.long()
            
            # Focus loss on medical tokens
            medical_loss = F.cross_entropy(
                medical_logits.view(-1, medical_logits.size(-1)),
                medical_labels.view(-1),
                ignore_index=0  # Ignore non-medical tokens
            )
        
        # Consistency regularization between client and server
        if server_logits is not None:
            consistency_loss = F.mse_loss(
                F.softmax(client_logits, dim=-1),
                F.softmax(server_logits, dim=-1)
            )
        
        # Add medical components to total loss
        total_loss = (loss_dict['total_loss'] + 
                     self.medical_weight * medical_loss + 
                     self.consistency_weight * consistency_loss)
        
        # Update loss dictionary
        loss_dict.update({
            'medical_loss': medical_loss,
            'consistency_loss': consistency_loss,
            'total_loss': total_loss
        })
        
        return loss_dict

class ContrastiveLoss(nn.Module):
    """
    Contrastive learning loss for better representation learning
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 1.0):
        """
        Args:
            temperature: Temperature parameter for contrastive loss
            margin: Margin for negative pairs
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, 
                client_features: torch.Tensor,
                server_features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between client and server features
        
        Args:
            client_features: Features from client model [batch_size, feature_dim]
            server_features: Features from server model [batch_size, feature_dim]
            labels: Labels for determining positive/negative pairs
            
        Returns:
            Contrastive loss
        """
        batch_size = client_features.size(0)
        
        # Normalize features
        client_features = F.normalize(client_features, dim=1)
        server_features = F.normalize(server_features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(client_features, server_features.T) / self.temperature
        
        # Create labels for positive pairs (same class)
        labels_expanded = labels.unsqueeze(0)
        positive_mask = torch.eq(labels_expanded, labels_expanded.T).float()
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        
        # Positive pairs
        positive_sim = exp_sim * positive_mask
        
        # All pairs (for normalization)
        all_sim = exp_sim.sum(dim=1, keepdim=True)
        
        # Contrastive loss
        loss = -torch.log(positive_sim.sum(dim=1) / all_sim.squeeze(1))
        
        return loss.mean()

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in medical QA
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AdversarialLoss(nn.Module):
    """
    Adversarial loss for privacy protection
    """
    
    def __init__(self, discriminator_weight: float = 0.1):
        """
        Args:
            discriminator_weight: Weight for adversarial loss component
        """
        super().__init__()
        self.discriminator_weight = discriminator_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, 
                discriminator_logits: torch.Tensor,
                is_real: bool = True) -> torch.Tensor:
        """
        Compute adversarial loss
        
        Args:
            discriminator_logits: Logits from discriminator
            is_real: Whether the input is real or generated
            
        Returns:
            Adversarial loss
        """
        batch_size = discriminator_logits.size(0)
        
        if is_real:
            real_labels = torch.ones(batch_size, 1, device=discriminator_logits.device)
            loss = self.bce_loss(discriminator_logits, real_labels)
        else:
            fake_labels = torch.zeros(batch_size, 1, device=discriminator_logits.device)
            loss = self.bce_loss(discriminator_logits, fake_labels)
        
        return self.discriminator_weight * loss

class CombinedFederatedLoss(nn.Module):
    """
    Combined loss function that incorporates multiple loss components
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary with loss weights
        """
        super().__init__()
        
        # Base distillation loss
        self.distillation_loss = FederatedKnowledgeDistillationLoss(
            temperature=config.get('temperature', 3.0),
            alpha=config.get('distillation_weight', 0.5),
            beta=config.get('task_weight', 0.5),
            label_smoothing=config.get('label_smoothing', 0.0)
        )
        
        # Additional loss components
        self.use_focal = config.get('use_focal_loss', False)
        if self.use_focal:
            self.focal_loss = FocalLoss(
                alpha=config.get('focal_alpha', 1.0),
                gamma=config.get('focal_gamma', 2.0)
            )
        
        self.use_contrastive = config.get('use_contrastive_loss', False)
        if self.use_contrastive:
            self.contrastive_loss = ContrastiveLoss(
                temperature=config.get('contrastive_temperature', 0.07)
            )
        
        # Loss weights
        self.focal_weight = config.get('focal_weight', 0.1)
        self.contrastive_weight = config.get('contrastive_weight', 0.1)
        
    def forward(self, 
                client_logits: torch.Tensor,
                server_logits: Optional[torch.Tensor],
                labels: torch.Tensor,
                client_features: Optional[torch.Tensor] = None,
                server_features: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute combined federated loss
        
        Returns:
            Dictionary with all loss components
        """
        # Base distillation loss
        loss_dict = self.distillation_loss(client_logits, server_logits, labels)
        
        total_loss = loss_dict['total_loss']
        
        # Add focal loss if enabled
        if self.use_focal:
            focal_loss_val = self.focal_loss(
                client_logits.view(-1, client_logits.size(-1)),
                labels.view(-1)
            )
            loss_dict['focal_loss'] = focal_loss_val
            total_loss += self.focal_weight * focal_loss_val
        
        # Add contrastive loss if enabled
        if self.use_contrastive and client_features is not None and server_features is not None:
            contrastive_loss_val = self.contrastive_loss(
                client_features, server_features, labels
            )
            loss_dict['contrastive_loss'] = contrastive_loss_val
            total_loss += self.contrastive_weight * contrastive_loss_val
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict

# Utility functions
def compute_perplexity(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute perplexity from logits and labels
    
    Args:
        logits: Predicted logits
        labels: Ground truth labels
        
    Returns:
        Perplexity value
    """
    cross_entropy = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction='mean'
    )
    perplexity = torch.exp(cross_entropy)
    return perplexity

def compute_kl_divergence(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between two distributions
    
    Args:
        logits1: First set of logits
        logits2: Second set of logits
        
    Returns:
        KL divergence value
    """
    prob1 = F.softmax(logits1, dim=-1)
    prob2 = F.softmax(logits2, dim=-1)
    
    kl_div = F.kl_div(
        F.log_softmax(logits1, dim=-1),
        prob2,
        reduction='batchmean'
    )
    
    return kl_div

# Test function
def test_loss_functions():
    """Test all loss functions"""
    batch_size, seq_len, vocab_size = 4, 10, 1000
    
    # Create sample data
    client_logits = torch.randn(batch_size, seq_len, vocab_size)
    server_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test distillation loss
    print("Testing Federated Knowledge Distillation Loss...")
    distill_loss = FederatedKnowledgeDistillationLoss()
    loss_dict = distill_loss(client_logits, server_logits, labels)
    
    print("Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Test medical QA loss
    print("\nTesting Medical QA Loss...")
    medical_loss = MedicalQALoss(distill_loss)
    medical_loss_dict = medical_loss(client_logits, server_logits, labels)
    
    print("Medical QA Loss components:")
    for key, value in medical_loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Test combined loss
    print("\nTesting Combined Federated Loss...")
    config = {
        'temperature': 3.0,
        'distillation_weight': 0.5,
        'task_weight': 0.5,
        'use_focal_loss': True,
        'use_contrastive_loss': True,
        'focal_weight': 0.1,
        'contrastive_weight': 0.1
    }
    
    combined_loss = CombinedFederatedLoss(config)
    
    # Create sample features for contrastive loss
    client_features = torch.randn(batch_size, 256)
    server_features = torch.randn(batch_size, 256)
    feature_labels = torch.randint(0, 10, (batch_size,))
    
    combined_loss_dict = combined_loss(
        client_logits, server_logits, labels,
        client_features=client_features,
        server_features=server_features
    )
    
    print("Combined Loss components:")
    for key, value in combined_loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Test utility functions
    print("\nTesting utility functions...")
    perplexity = compute_perplexity(client_logits, labels)
    kl_div = compute_kl_divergence(client_logits, server_logits)
    
    print(f"Perplexity: {perplexity.item():.4f}")
    print(f"KL Divergence: {kl_div.item():.4f}")
    
    print("\nAll loss function tests completed successfully!")

if __name__ == "__main__":
    test_loss_functions()
