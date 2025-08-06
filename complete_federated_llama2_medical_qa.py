
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer
)
from datasets import load_dataset
import numpy as np
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Tuple
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('federated_llama2_medical_qa.log')
    ]
)
logger = logging.getLogger(__name__)

def _make_causal_mask(input_shape: Tuple[int, int], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Create a causal mask for LLaMA-2 attention mechanism.
    Returns a mask of shape (batch_size, 1, seq_length, seq_length).
    """
    batch_size, seq_length = input_shape
    mask = torch.zeros((batch_size, 1, seq_length, seq_length), dtype=dtype, device=device)
    mask = mask.masked_fill(
        torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool, device=device), diagonal=1).unsqueeze(0).unsqueeze(0),
        float('-inf')
    )
    return mask

class MedicalQADataset(Dataset):
    """
    Real Medical QA Dataset using PubMedQA, MedQA, and MedMCQA
    """
    
    def __init__(self, tokenizer, max_length: int = 512, max_samples: int = 10000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        logger.info("Loading REAL medical QA datasets...")
        self._load_real_datasets(max_samples)
        
        logger.info(f"✅ Loaded {len(self.data)} real medical QA samples")
    
    def _load_real_datasets(self, max_samples: int):
        """Load real medical datasets"""
        
        # 1. Load PubMedQA
        try:
            logger.info("Loading PubMedQA...")
            pubmed_qa = load_dataset("pubmed_qa", "pqa_labeled", split="train")
            count = 0
            for item in pubmed_qa:
                if count >= max_samples // 3:
                    break
                if item['final_decision'] != 'maybe':
                    context = ' '.join(item['context']['contexts'])
                    self.data.append({
                        'question': item['question'],
                        'answer': item['long_answer'] if item['long_answer'] else f"The answer is {item['final_decision']}.",
                        'context': context,
                        'source': 'PubMedQA'
                    })
                    count += 1
            logger.info(f"✅ PubMedQA: {count} samples")
        except Exception as e:
            logger.error(f"Failed to load PubMedQA: {e}")
        
        # 2. Load MedQA (using bigbio/med_qa)
        try:
            logger.info("Loading MedQA...")
            medqa = load_dataset("bigbio/med_qa", split="train")
            count = 0
            for item in medqa:
                if count >= max_samples // 3:
                    break
                question = item['question']
                if 'options' in item and item['options']:
                    options_text = "\n".join([f"{k}: {v}" for k, v in item['options'].items()])
                    question = f"{question}\n\nOptions:\n{options_text}"
                
                self.data.append({
                    'question': question,
                    'answer': item['answer'][0] if isinstance(item['answer'], list) else item['answer'],
                    'source': 'MedQA'
                })
                count += 1
            logger.info(f"✅ MedQA: {count} samples")
        except Exception as e:
            logger.error(f"Failed to load MedQA: {e}")
        
        # 3. Load MedMCQA
        try:
            logger.info("Loading MedMCQA...")
            medmcqa = load_dataset("medmcqa", split="train")
            count = 0
            for item in medmcqa:
                if count >= max_samples // 3:
                    break
                
                options = [item['opa'], item['opb'], item['opc'], item['opd']]
                correct_idx = item['cop']
                correct_answer = options[correct_idx]
                
                question = item['question']
                options_text = "\n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(options)])
                formatted_question = f"{question}\n\n{options_text}"
                
                explanation = item.get('exp', '')
                if explanation:
                    answer = f"The correct answer is {chr(65+correct_idx)}: {correct_answer}\n\nExplanation: {explanation}"
                else:
                    answer = f"The correct answer is {chr(65+correct_idx)}: {correct_answer}"
                
                self.data.append({
                    'question': formatted_question,
                    'answer': answer,
                    'source': 'MedMCQA'
                })
                count += 1
            logger.info(f"✅ MedMCQA: {count} samples")
        except Exception as e:
            logger.error(f"Failed to load MedMCQA: {e}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format for medical QA
        question = item['question']
        answer = item['answer']
        
        # Create training format
        full_text = f"Question: {question}\nAnswer: {answer}"
        prompt_text = f"Question: {question}\nAnswer:"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        prompt_encoding = self.tokenizer(
            prompt_text,
            max_length=self.max_length//2,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (mask question part)
        labels = encoding['input_ids'].clone()
        prompt_length = len(prompt_encoding['input_ids'][0])
        labels[0, :prompt_length] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'question': question,
            'answer': answer,
            'source': item['source']
        }

class LLaMAMedicalClient(nn.Module):
    """
    REAL LLaMA-2-7B Client Model
    Uses first N layers of LLaMA-2-7B
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", 
                 split_layer: int = 16, device: str = 'cuda'):
        super().__init__()
        
        self.model_name = model_name
        self.split_layer = split_layer
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16  # Match model's dtype
        
        logger.info(f"🦙 Loading REAL LLaMA-2-7B Client: {model_name} on {self.device} with dtype {self.dtype}")
        
        # Load REAL LLaMA-2-7B model
        self.full_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.config = self.full_model.config
        
        # Extract client components
        self.embed_tokens = self.full_model.model.embed_tokens
        
        # Extract first split_layer transformer layers
        self.layers = nn.ModuleList()
        for i in range(self.split_layer):
            self.layers.append(self.full_model.model.layers[i])
        
        # Create LayerNorm with matching dtype
        self.norm = nn.LayerNorm(self.config.hidden_size, dtype=self.dtype).to(self.device)
        
        # Move all parameters to the specified device
        self.to(self.device)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"✅ LLaMA-2 Client loaded: {total_params/1e6:.1f}M parameters")
        logger.info(f"✅ Client norm layer device: {next(self.norm.parameters()).device}")
        logger.info(f"✅ Client norm layer dtype: {next(self.norm.parameters()).dtype}")
    
    def forward(self, input_ids, attention_mask=None, position_ids=None):
        # Ensure inputs are on the correct device and dtype
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Embedding
        hidden_states = self.embed_tokens(input_ids)
        logger.debug(f"Hidden states dtype after embed: {hidden_states.dtype}")
        
        # Create 4D causal attention mask
        if attention_mask is not None:
            batch_size, seq_length = input_ids.shape
            # Generate causal mask
            causal_mask = _make_causal_mask(
                input_shape=(batch_size, seq_length),
                dtype=self.dtype,  # Use model's dtype
                device=hidden_states.device
            )
            # Combine with input attention mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # Shape: (batch_size, 1, 1, seq_length)
            attention_mask = causal_mask + attention_mask  # Combine masks (additive for -inf)
        
        # Apply client layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False
            )
            hidden_states = layer_outputs[0]
        
        # Layer normalization
        hidden_states = self.norm(hidden_states)
        logger.debug(f"Hidden states dtype after norm: {hidden_states.dtype}")
        return hidden_states
    
    def get_head_parameters(self):
        """Get parameters for federated averaging"""
        return {
            'embed_tokens.weight': self.embed_tokens.weight.clone().detach(),
            'norm.weight': self.norm.weight.clone().detach(),
            'norm.bias': self.norm.bias.clone().detach() if hasattr(self.norm, 'bias') and self.norm.bias is not None else None
        }
    
    def set_head_parameters(self, params):
        """Set parameters from federated averaging"""
        if 'embed_tokens.weight' in params:
            self.embed_tokens.weight.data.copy_(params['embed_tokens.weight'].to(self.device).to(self.dtype))
        if 'norm.weight' in params:
            self.norm.weight.data.copy_(params['norm.weight'].to(self.device).to(self.dtype))
        if 'norm.bias' in params and params['norm.bias'] is not None:
            if hasattr(self.norm, 'bias') and self.norm.bias is not None:
                self.norm.bias.data.copy_(params['norm.bias'].to(self.device).to(self.dtype))

class LLaMAMedicalServer(nn.Module):
    """
    REAL LLaMA-2-7B Server Model
    Uses remaining layers of LLaMA-2-7B + LM head
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", 
                 split_layer: int = 16, device: str = 'cuda'):
        super().__init__()
        
        self.model_name = model_name
        self.split_layer = split_layer
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16  # Match model's dtype
        
        logger.info(f"🦙 Loading REAL LLaMA-2-7B Server: {model_name} on {self.device} with dtype {self.dtype}")
        
        # Load REAL LLaMA-2-7B model
        self.full_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.config = self.full_model.config
        
        # Extract server components (remaining layers)
        self.layers = nn.ModuleList()
        total_layers = len(self.full_model.model.layers)
        
        for i in range(self.split_layer, total_layers):
            self.layers.append(self.full_model.model.layers[i])
        
        # Final components
        self.norm = self.full_model.model.norm.to(self.device)
        self.lm_head = self.full_model.lm_head.to(self.device)
        
        # Move all parameters to the specified device
        self.to(self.device)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"✅ LLaMA-2 Server loaded: {total_params/1e6:.1f}M parameters")
        logger.info(f"✅ Server norm layer device: {next(self.norm.parameters()).device}")
        logger.info(f"✅ Server norm layer dtype: {next(self.norm.parameters()).dtype}")
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # Ensure inputs are on the correct device
        hidden_states = hidden_states.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Create 4D causal attention mask
        if attention_mask is not None:
            batch_size, seq_length = hidden_states.shape[:2]
            # Generate causal mask
            causal_mask = _make_causal_mask(
                input_shape=(batch_size, seq_length),
                dtype=self.dtype,
                device=hidden_states.device
            )
            # Combine with input attention mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # Shape: (batch_size, 1, 1, seq_length)
            attention_mask = causal_mask + attention_mask  # Combine masks (additive for -inf)
        
        # Apply server layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False
            )
            hidden_states = layer_outputs[0]
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        return logits

class PrivacyMechanism:
    """Privacy mechanisms: Gaussian noise + quantization"""
    
    def __init__(self, sigma: float = 0.1, quantization_bits: int = 8):
        self.sigma = sigma
        self.quantization_bits = quantization_bits
    
    def add_noise_and_quantize(self, data: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise and quantize"""
        # Add Gaussian noise
        if self.sigma > 0:
            noise = torch.randn_like(data) * self.sigma
            noisy_data = data + noise
        else:
            noisy_data = data
        
        # Simple quantization
        if self.quantization_bits < 32:
            # Scale to quantization range
            data_min, data_max = noisy_data.min(), noisy_data.max()
            if data_max > data_min:
                max_val = 2**(self.quantization_bits - 1) - 1
                min_val = -2**(self.quantization_bits - 1)
                
                scaled = (noisy_data - data_min) / (data_max - data_min)
                quantized = torch.round(scaled * (max_val - min_val) + min_val)
                reconstructed = (quantized - min_val) / (max_val - min_val) * (data_max - data_min) + data_min
                return reconstructed
        
        return noisy_data

class MedicalQAMetrics:
    """Medical QA evaluation metrics"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def compute_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute token-level accuracy"""
        pred_tokens = torch.argmax(predictions, dim=-1)
        mask = (labels != -100)
        
        if mask.sum() > 0:
            correct = (pred_tokens[mask] == labels[mask]).float()
            return correct.mean().item()
        return 0.0
    
    def compute_perplexity(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute perplexity"""
        loss = nn.CrossEntropyLoss(ignore_index=-100)(
            predictions.view(-1, predictions.size(-1)),
            labels.view(-1)
        )
        return torch.exp(loss).item()

class CompleteFederatedTrainer:
    """
    Complete Federated Trainer for LLaMA-2-7B Medical QA
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("🚀 Initializing Complete Federated LLaMA-2-7B Medical QA")
        logger.info(f"Device: {self.device}")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU Memory: {gpu_memory:.1f}GB")
            if gpu_memory < 14:
                logger.warning("⚠️ May need more GPU memory for LLaMA-2-7B")
        
        # Initialize components
        self._setup_tokenizer()
        self._setup_privacy()
        self._setup_models()
        self._setup_data()
        self._setup_training()
        
        # Training state
        self.current_round = 0
        self.best_accuracy = 0.0
    
    def _setup_tokenizer(self):
        """Setup tokenizer"""
        model_name = self.config['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"✅ Tokenizer loaded: {model_name}")
    
    def _setup_privacy(self):
        """Setup privacy mechanisms"""
        privacy_config = self.config.get('privacy', {})
        self.privacy_mechanism = PrivacyMechanism(
            sigma=privacy_config.get('sigma', 0.1),
            quantization_bits=privacy_config.get('quantization_bits', 8)
        )
        logger.info(f"✅ Privacy: σ={self.privacy_mechanism.sigma}, {self.privacy_mechanism.quantization_bits}-bit")
    
    def _setup_models(self):
        """Setup client and server models"""
        model_name = self.config['model_name']
        split_layer = self.config['split_layer']
        num_clients = self.config['num_clients']
        
        # Create clients
        self.clients = []
        for i in range(num_clients):
            client = LLaMAMedicalClient(model_name, split_layer, self.device)
            self.clients.append(client)
        
        # Create server
        self.server = LLaMAMedicalServer(model_name, split_layer, self.device)
        
        logger.info(f"✅ Models: {num_clients} clients + 1 server (split at layer {split_layer})")
    
    def _setup_data(self):
        """Setup federated data"""
        max_samples = self.config.get('max_samples', 10000)
        max_length = self.config.get('max_length', 512)
        batch_size = self.config.get('batch_size', 2)
        
        # Create dataset
        full_dataset = MedicalQADataset(
            tokenizer=self.tokenizer,
            max_length=max_length,
            max_samples=max_samples
        )
        
        # Train/test split
        total_size = len(full_dataset)
        test_size = int(0.2 * total_size)
        train_size = total_size - test_size
        
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        
        # Federated client splits
        num_clients = self.config['num_clients']
        client_sizes = [train_size // num_clients] * num_clients
        
        # Distribute remainder
        for i in range(train_size % num_clients):
            client_sizes[i] += 1
        
        client_datasets = random_split(train_dataset, client_sizes)
        
        # Create data loaders
        self.client_loaders = []
        for dataset in client_datasets:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
            )
            self.client_loaders.append(loader)
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"✅ Data: {client_sizes} samples per client, {len(test_dataset)} test samples")
    
    def _setup_training(self):
        """Setup optimizers and loss"""
        lr = self.config.get('learning_rate', 5e-5)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        # Client optimizers
        self.client_optimizers = []
        for client in self.clients:
            optimizer = optim.AdamW(client.parameters(), lr=lr, weight_decay=weight_decay)
            self.client_optimizers.append(optimizer)
        
        # Server optimizer
        self.server_optimizer = optim.AdamW(self.server.parameters(), lr=lr*0.5, weight_decay=weight_decay)
        
        # Loss and metrics
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics = MedicalQAMetrics(self.tokenizer)
        
        logger.info(f"✅ Training setup: LR={lr}, clients={len(self.clients)}")
    
    def train_client_locally(self, client, server, dataloader, optimizer, local_epochs=1):
        """Train a client locally"""
        client.train()
        server.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(local_epochs):
            for batch in tqdm(dataloader, desc=f"Local Epoch {epoch+1}", leave=False):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Client forward pass
                client_hidden = client(input_ids, attention_mask)
                
                # Apply privacy mechanisms
                private_hidden = self.privacy_mechanism.add_noise_and_quantize(client_hidden)
                
                # Server forward pass
                server_logits = server(private_hidden, attention_mask)
                
                # Compute loss
                loss = self.loss_fn(
                    server_logits.view(-1, server_logits.size(-1)),
                    labels.view(-1)
                )
                
                # Backward pass
                optimizer.zero_grad()
                self.server_optimizer.zero_grad()
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(client.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(server.parameters(), 1.0)
                
                optimizer.step()
                self.server_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def federated_averaging(self):
        """Perform federated averaging"""
        if len(self.clients) <= 1:
            return
        
        # Collect head parameters
        client_params = []
        for client in self.clients:
            params = client.get_head_parameters()
            client_params.append(params)
        
        # Average parameters
        averaged_params = {}
        for key in client_params[0].keys():
            if client_params[0][key] is not None:
                param_tensors = [params[key] for params in client_params if params[key] is not None]
                if param_tensors:
                    averaged_params[key] = torch.stack(param_tensors).mean(dim=0)
        
        # Distribute back to clients
        for client in self.clients:
            client.set_head_parameters(averaged_params)
        
        logger.info("✅ Federated averaging completed")
    
    def evaluate(self):
        """Evaluate on test set"""
        client = self.clients[0]  # Use first client as representative
        client.eval()
        self.server.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_perplexity = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                client_hidden = client(input_ids, attention_mask)
                server_logits = self.server(client_hidden, attention_mask)
                
                # Compute metrics
                loss = self.loss_fn(
                    server_logits.view(-1, server_logits.size(-1)),
                    labels.view(-1)
                )
                
                accuracy = self.metrics.compute_accuracy(server_logits, labels)
                perplexity = self.metrics.compute_perplexity(server_logits, labels)
                
                total_loss += loss.item()
                total_accuracy += accuracy
                total_perplexity += perplexity
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_perplexity = total_perplexity / num_batches
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'perplexity': avg_perplexity
        }
    
    def save_best_model(self):
        """Save the best model"""
        save_dir = Path("./results/best_llama2_medical_qa")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        torch.save(self.clients[0].state_dict(), save_dir / "client_model.pt")
        torch.save(self.server.state_dict(), save_dir / "server_model.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        # Save config
        with open(save_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"✅ Best model saved to {save_dir}")
    
    def train_federated(self):
        """Main federated training loop"""
        num_rounds = self.config['num_rounds']
        local_epochs = self.config.get('local_epochs', 1)
        eval_frequency = self.config.get('eval_frequency', 2)
        
        logger.info(f"🚀 Starting federated training: {num_rounds} rounds")
        
        training_start = time.time()
        
        for round_num in range(1, num_rounds + 1):
            round_start = time.time()
            self.current_round = round_num
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 ROUND {round_num}/{num_rounds}")
            logger.info(f"{'='*60}")
            
            # Train each client locally
            round_losses = []
            for client_id, (client, loader, optimizer) in enumerate(zip(
                self.clients, self.client_loaders, self.client_optimizers
            )):
                logger.info(f"Training Client {client_id}...")
                
                client_loss = self.train_client_locally(
                    client, self.server, loader, optimizer, local_epochs
                )
                round_losses.append(client_loss)
                
                logger.info(f"  Client {client_id} Loss: {client_loss:.4f}")
            
            # Federated averaging
            self.federated_averaging()
            
            # Evaluation
            if round_num % eval_frequency == 0:
                eval_results = self.evaluate()
                
                logger.info(f"📊 Round {round_num} Evaluation:")
                logger.info(f"  Loss: {eval_results['loss']:.4f}")
                logger.info(f"  Accuracy: {eval_results['accuracy']:.4f}")
                logger.info(f"  Perplexity: {eval_results['perplexity']:.2f}")
                
                # Save best model
                if eval_results['accuracy'] > self.best_accuracy:
                    self.best_accuracy = eval_results['accuracy']
                    self.save_best_model()
                    logger.info(f"💾 New best model! Accuracy: {self.best_accuracy:.4f}")
            
            round_time = time.time() - round_start
            logger.info(f"⏱️ Round {round_num} completed in {round_time:.1f}s")
        
        total_time = time.time() - training_start
        
        logger.info(f"\n🎉 TRAINING COMPLETED!")
        logger.info(f"  Total time: {total_time/3600:.2f} hours")
        logger.info(f"  Best accuracy: {self.best_accuracy:.4f}")
        logger.info(f"  Rounds completed: {num_rounds}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Federated LLaMA-2-7B Medical QA")
    parser.add_argument('--model_name', default='meta-llama/Llama-2-7b-hf', help='Model name')
    parser.add_argument('--num_clients', type=int, default=3, help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=10, help='Training rounds')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--max_samples', type=int, default=10000, help='Max samples')
    parser.add_argument('--split_layer', type=int, default=16, help='Split layer')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'model_name': args.model_name,
        'split_layer': args.split_layer,
        'num_clients': args.num_clients,
        'num_rounds': args.num_rounds,
        'local_epochs': 1,
        'batch_size': args.batch_size,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'max_length': 512,
        'max_samples': args.max_samples,
        'eval_frequency': 2,
        'privacy': {
            'sigma': 0.1,
            'quantization_bits': 8
        }
    }
    
    logger.info("🦙 COMPLETE LLaMA-2-7B FEDERATED MEDICAL QA")
    logger.info("📋 Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Check GPU
    if not torch.cuda.is_available():
        logger.error("❌ CUDA required for LLaMA-2-7B!")
        return
    
    # Estimate training time
    total_samples = config['max_samples']
    samples_per_client = total_samples // config['num_clients']
    batches_per_client = samples_per_client // config['batch_size']
    total_batches = batches_per_client * config['num_clients'] * config['num_rounds']
    estimated_hours = total_batches * 0.5 / 3600  # ~0.5 seconds per batch
    
    logger.info(f"⏱️ ESTIMATED TRAINING TIME: {estimated_hours:.1f} hours")
    logger.info(f"  ({total_batches} total batches)")
    logger.info(f"  ({samples_per_client} samples per client)")
    
    try:
        # Create and run trainer
        trainer = CompleteFederatedTrainer(config)
        trainer.train_federated()
        
        logger.info("🎉 SUCCESS! Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ TRAINING FAILED: {e}")
        logger.error("💡 Common solutions:")
        logger.error("  1. Request LLaMA-2 access: https://huggingface.co/meta-llama/Llama-2-7b-hf")
        logger.error("  2. Login: huggingface-cli login")
        logger.error("  3. Check GPU memory (need 16GB+)")
        logger.error("  4. Install requirements: pip install torch transformers datasets accelerate bitsandbytes")
        raise

if __name__ == "__main__":
    main()
