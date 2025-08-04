"""
LLaMA Split Model Implementation for Federated Learning
Based on the methodology described in the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, Tuple, Optional
import math

class LLaMAEmbedding(nn.Module):
    """Head component - Embedding layer for clients"""
    
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_length: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        embeddings = token_embeds + pos_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class LLaMATransformerBlock(nn.Module):
    """Single transformer block for LLaMA"""
    
    def __init__(self, embed_dim: int, num_heads: int = 16, ff_dim: int = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or 4 * embed_dim
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, self.ff_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.ff_dim, embed_dim),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        x = self.norm1(x + attn_out)
        
        # Feed forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class ClientInitialLayers(nn.Module):
    """Initial processing layers for global model path (Layers 1-5)"""
    
    def __init__(self, embed_dim: int, num_layers: int = 5, num_heads: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([
            LLaMATransformerBlock(embed_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x

class ClientLocalBody(nn.Module):
    """Local body component - Distilled layers (2-3 layers)"""
    
    def __init__(self, embed_dim: int, num_layers: int = 3, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            LLaMATransformerBlock(embed_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x

class ClientTail(nn.Module):
    """Task-specific tail for clients"""
    
    def __init__(self, embed_dim: int, vocab_size: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            LLaMATransformerBlock(embed_dim, num_heads=8) 
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Project to vocabulary space
        logits = self.output_projection(x)
        return logits

class ServerGlobalBody(nn.Module):
    """Server global body (Layers 6-24)"""
    
    def __init__(self, embed_dim: int, num_layers: int = 18, num_heads: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([
            LLaMATransformerBlock(embed_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x

class ServerTail(nn.Module):
    """Server task-specific tail"""
    
    def __init__(self, embed_dim: int, vocab_size: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            LLaMATransformerBlock(embed_dim, num_heads=8) 
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Project to vocabulary space  
        logits = self.output_projection(x)
        return logits

class LLaMAClientModel(nn.Module):
    """Complete client model with dual paths"""
    
    def __init__(self, config: Dict):
        super().__init__()
        embed_dim = config['embed_dim']
        vocab_size = config['vocab_size']
        max_seq_length = config['max_seq_length']
        
        # Shared head (embedding layer)
        self.head = LLaMAEmbedding(vocab_size, embed_dim, max_seq_length)
        
        # Global model path - initial layers
        self.initial_layers = ClientInitialLayers(
            embed_dim, 
            num_layers=config.get('initial_layers', 5)
        )
        
        # Local model path
        self.local_body = ClientLocalBody(
            embed_dim,
            num_layers=config.get('local_body_layers', 3)
        )
        self.local_tail = ClientTail(embed_dim, vocab_size)
        
    def forward_local(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Local model forward pass"""
        # Head processing
        embeddings = self.head(input_ids)
        
        # Local body processing
        local_features = self.local_body(embeddings, attention_mask)
        
        # Local tail processing
        local_logits = self.local_tail(local_features, attention_mask)
        
        return local_logits
    
    def forward_global_client_side(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Global model forward pass - client side only"""
        # Head processing
        embeddings = self.head(input_ids)
        
        # Initial layers processing (to be sent to server)
        smashed_data = self.initial_layers(embeddings, attention_mask)
        
        return smashed_data
    
    def get_head_parameters(self) -> Dict[str, torch.Tensor]:
        """Get head parameters for federated averaging"""
        return {name: param.clone() for name, param in self.head.named_parameters()}
    
    def set_head_parameters(self, head_params: Dict[str, torch.Tensor]):
        """Set head parameters from federated averaging"""
        for name, param in self.head.named_parameters():
            if name in head_params:
                param.data.copy_(head_params[name])

class LLaMAServerModel(nn.Module):
    """Server model with global body and tail"""
    
    def __init__(self, config: Dict):
        super().__init__()
        embed_dim = config['embed_dim']
        vocab_size = config['vocab_size']
        
        # Global body (layers 6-24)
        self.global_body = ServerGlobalBody(
            embed_dim,
            num_layers=config.get('server_body_layers', 18)
        )
        
        # Task-specific tail
        self.tail = ServerTail(embed_dim, vocab_size)
        
        # Storage for aggregated head weights
        self.head_weights = None
        
    def forward(self, smashed_data: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Server forward pass"""
        # Global body processing
        global_features = self.global_body(smashed_data, attention_mask)
        
        # Tail processing
        server_logits = self.tail(global_features, attention_mask)
        
        return server_logits
    
    def update_head_weights(self, aggregated_weights: Dict[str, torch.Tensor]):
        """Update aggregated head weights"""
        self.head_weights = aggregated_weights

def create_llama_models(config: Dict) -> Tuple[LLaMAClientModel, LLaMAServerModel]:
    """Factory function to create client and server models"""
    
    # Model configuration
    model_config = {
        'embed_dim': config.get('embed_dim', 1024),
        'vocab_size': config.get('vocab_size', 50257),
        'max_seq_length': config.get('max_seq_length', 512),
        'initial_layers': config.get('split_layer', 5),
        'local_body_layers': config.get('client_local_body_layers', 3),
        'server_body_layers': config.get('server_body_layers', 18)
    }
    
    client_model = LLaMAClientModel(model_config)
    server_model = LLaMAServerModel(model_config)
    
    return client_model, server_model

def load_pretrained_weights(client_model: LLaMAClientModel, server_model: LLaMAServerModel, 
                          model_name: str = "microsoft/DialoGPT-medium"):
    """Load pretrained weights and split them between client and server"""
    
    try:
        # Load pretrained model
        pretrained_model = AutoModel.from_pretrained(model_name)
        pretrained_dict = pretrained_model.state_dict()
        
        # Initialize client model head with pretrained embeddings
        if 'embeddings.word_embeddings.weight' in pretrained_dict:
            client_model.head.token_embedding.weight.data.copy_(
                pretrained_dict['embeddings.word_embeddings.weight']
            )
            
        # Initialize position embeddings if available
        if 'embeddings.position_embeddings.weight' in pretrained_dict:
            client_model.head.position_embedding.weight.data.copy_(
                pretrained_dict['embeddings.position_embeddings.weight']
            )
            
        print(f"Successfully loaded pretrained weights from {model_name}")
        
    except Exception as e:
        print(f"Warning: Could not load pretrained weights: {e}")
        print("Using random initialization instead")

if __name__ == "__main__":
    # Test model creation
    config = {
        'embed_dim': 1024,
        'vocab_size': 50257,
        'max_seq_length': 512,
        'split_layer': 5,
        'client_local_body_layers': 3,
        'server_body_layers': 18
    }
    
    client_model, server_model = create_llama_models(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    # Client local forward
    local_logits = client_model.forward_local(input_ids)
    print(f"Local logits shape: {local_logits.shape}")
    
    # Client global forward (smashed data)
    smashed_data = client_model.forward_global_client_side(input_ids)
    print(f"Smashed data shape: {smashed_data.shape}")
    
    # Server forward
    server_logits = server_model.forward(smashed_data)
    print(f"Server logits shape: {server_logits.shape}")
    
    print("Model architecture test completed successfully!")
