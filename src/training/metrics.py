"""
Evaluation Metrics for Federated Medical QA
Implements comprehensive evaluation metrics including medical-specific measures
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import math
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import re

class MedicalQAMetrics:
    """
    Comprehensive metrics for medical question answering evaluation
    """
    
    def __init__(self, tokenizer=None):
        """
        Args:
            tokenizer: Tokenizer for text processing (optional)
        """
        self.tokenizer = tokenizer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1
        
        # Medical-specific vocabulary for domain evaluation
        self.medical_terms = self._load_medical_vocabulary()
        
    def _load_medical_vocabulary(self) -> set:
        """Load medical vocabulary for domain-specific evaluation"""
        # Basic medical terms - in practice, this would be loaded from a comprehensive medical ontology
        medical_terms = {
            'diagnosis', 'treatment', 'symptom', 'disease', 'patient', 'therapy',
            'medication', 'prescription', 'dosage', 'side effect', 'adverse reaction',
            'clinical', 'pathology', 'anatomy', 'physiology', 'pharmacology',
            'hypertension', 'diabetes', 'cancer', 'infection', 'inflammation',
            'pneumonia', 'asthma', 'arthritis', 'migraine', 'depression',
            'anxiety', 'heart disease', 'stroke', 'kidney', 'liver',
            'blood pressure', 'cholesterol', 'glucose', 'insulin', 'antibiotic'
        }
        return medical_terms
    
    def compute_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute classification accuracy
        
        Args:
            predictions: Predicted probabilities [batch_size, seq_len, vocab_size]
            labels: Ground truth labels [batch_size, seq_len]
            
        Returns:
            Accuracy score
        """
        # Get predicted classes
        pred_classes = torch.argmax(predictions, dim=-1)
        
        # Flatten tensors
        pred_flat = pred_classes.view(-1)
        labels_flat = labels.view(-1)
        
        # Compute accuracy (ignoring padding tokens if label is 0)
        mask = labels_flat != 0  # Assuming 0 is padding token
        if mask.sum() > 0:
            correct = (pred_flat[mask] == labels_flat[mask]).float()
            accuracy = correct.mean().item()
        else:
            accuracy = 0.0
        
        return accuracy
    
    def compute_top_k_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
        """
        Compute top-k accuracy
        
        Args:
            predictions: Predicted probabilities
            labels: Ground truth labels
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy
        """
        # Get top-k predictions
        _, top_k_preds = torch.topk(predictions, k, dim=-1)
        
        # Expand labels to match top-k shape
        labels_expanded = labels.unsqueeze(-1).expand_as(top_k_preds)
        
        # Check if true label is in top-k
        correct = (top_k_preds == labels_expanded).any(dim=-1)
        
        # Compute accuracy (ignoring padding)
        mask = labels != 0
        if mask.sum() > 0:
            top_k_acc = correct[mask].float().mean().item()
        else:
            top_k_acc = 0.0
        
        return top_k_acc
    
    def compute_perplexity(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute perplexity
        
        Args:
            predictions: Predicted probabilities
            labels: Ground truth labels
            
        Returns:
            Perplexity score
        """
        # Compute cross-entropy loss
        log_probs = torch.log(predictions + 1e-8)  # Add small epsilon for numerical stability
        
        # Flatten tensors
        log_probs_flat = log_probs.view(-1, log_probs.size(-1))
        labels_flat = labels.view(-1)
        
        # Compute negative log-likelihood
        nll = F.nll_loss(log_probs_flat, labels_flat, ignore_index=0, reduction='mean')
        
        # Compute perplexity
        perplexity = torch.exp(nll).item()
        
        return perplexity
    
    def compute_f1_score(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute precision, recall, and F1 score
        
        Args:
            predictions: Predicted probabilities
            labels: Ground truth labels
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        # Get predicted classes
        pred_classes = torch.argmax(predictions, dim=-1)
        
        # Flatten and convert to numpy
        pred_flat = pred_classes.view(-1).cpu().numpy()
        labels_flat = labels.view(-1).cpu().numpy()
        
        # Remove padding tokens
        mask = labels_flat != 0
        pred_filtered = pred_flat[mask]
        labels_filtered = labels_flat[mask]
        
        if len(pred_filtered) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_filtered, pred_filtered, average='weighted', zero_division=0
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def compute_bleu_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute BLEU scores for text generation
        
        Args:
            predictions: List of predicted text strings
            references: List of reference text strings
            
        Returns:
            Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
        """
        if not predictions or not references:
            return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0}
        
        bleu_scores = {'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': []}
        
        for pred, ref in zip(predictions, references):
            # Tokenize
            pred_tokens = pred.lower().split()
            ref_tokens = [ref.lower().split()]  # BLEU expects list of reference lists
            
            # Compute BLEU scores
            try:
                bleu1 = sentence_bleu(ref_tokens, pred_tokens, weights=(1.0, 0, 0, 0), 
                                    smoothing_function=self.smoothing_function)
                bleu2 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0),
                                    smoothing_function=self.smoothing_function)
                bleu3 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0),
                                    smoothing_function=self.smoothing_function)
                bleu4 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                                    smoothing_function=self.smoothing_function)
                
                bleu_scores['bleu1'].append(bleu1)
                bleu_scores['bleu2'].append(bleu2)
                bleu_scores['bleu3'].append(bleu3)
                bleu_scores['bleu4'].append(bleu4)
            except:
                # Handle edge cases
                bleu_scores['bleu1'].append(0.0)
                bleu_scores['bleu2'].append(0.0)
                bleu_scores['bleu3'].append(0.0)
                bleu_scores['bleu4'].append(0.0)
        
        # Average scores
        avg_bleu_scores = {}
        for key, scores in bleu_scores.items():
            avg_bleu_scores[key] = np.mean(scores) if scores else 0.0
        
        return avg_bleu_scores
    
    def compute_rouge_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores for text generation
        
        Args:
            predictions: List of predicted text strings
            references: List of reference text strings
            
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        if not predictions or not references:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            try:
                scores = self.rouge_scorer.score(ref, pred)
                rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
            except:
                rouge_scores['rouge1'].append(0.0)
                rouge_scores['rouge2'].append(0.0)
                rouge_scores['rougeL'].append(0.0)
        
        # Average scores
        avg_rouge_scores = {}
        for key, scores in rouge_scores.items():
            avg_rouge_scores[key] = np.mean(scores) if scores else 0.0
        
        return avg_rouge_scores
    
    def compute_medical_terminology_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute accuracy specifically for medical terminology
        
        Args:
            predictions: List of predicted text strings
            references: List of reference text strings
            
        Returns:
            Medical terminology accuracy
        """
        if not predictions or not references:
            return 0.0
        
        medical_accuracies = []
        
        for pred, ref in zip(predictions, references):
            # Extract medical terms from prediction and reference
            pred_medical_terms = self._extract_medical_terms(pred)
            ref_medical_terms = self._extract_medical_terms(ref)
            
            if not ref_medical_terms:
                # No medical terms in reference, skip
                continue
            
            # Compute accuracy
            correct_terms = pred_medical_terms.intersection(ref_medical_terms)
            accuracy = len(correct_terms) / len(ref_medical_terms)
            medical_accuracies.append(accuracy)
        
        return np.mean(medical_accuracies) if medical_accuracies else 0.0
    
    def _extract_medical_terms(self, text: str) -> set:
        """Extract medical terms from text"""
        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Find medical terms
        medical_terms_found = set()
        for word in words:
            if word in self.medical_terms:
                medical_terms_found.add(word)
        
        return medical_terms_found
    
    def compute_semantic_similarity(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute semantic similarity between predictions and references
        This is a simplified version - in practice, would use embeddings
        
        Args:
            predictions: List of predicted text strings
            references: List of reference text strings
            
        Returns:
            Average semantic similarity score
        """
        if not predictions or not references:
            return 0.0
        
        similarities = []
        
        for pred, ref in zip(predictions, references):
            # Simple word overlap similarity (Jaccard similarity)
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            if not ref_words:
                similarities.append(0.0)
                continue
            
            intersection = pred_words.intersection(ref_words)
            union = pred_words.union(ref_words)
            
            similarity = len(intersection) / len(union) if union else 0.0
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def compute_batch_metrics(self, 
                            predictions: torch.Tensor,
                            labels: torch.Tensor,
                            input_ids: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compute metrics for a single batch
        
        Args:
            predictions: Predicted probabilities
            labels: Ground truth labels
            input_ids: Input token IDs (optional, for text generation metrics)
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = self.compute_accuracy(predictions, labels)
        metrics['top5_accuracy'] = self.compute_top_k_accuracy(predictions, labels, k=5)
        metrics['perplexity'] = self.compute_perplexity(predictions, labels)
        
        # F1 metrics
        f1_metrics = self.compute_f1_score(predictions, labels)
        metrics.update(f1_metrics)
        
        return metrics
    
    def compute_comprehensive_metrics(self,
                                    predictions: torch.Tensor,
                                    labels: torch.Tensor,
                                    input_ids: Optional[torch.Tensor] = None,
                                    predicted_texts: Optional[List[str]] = None,
                                    reference_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics
        
        Args:
            predictions: Predicted probabilities
            labels: Ground truth labels
            input_ids: Input token IDs (optional)
            predicted_texts: List of predicted text strings (optional)
            reference_texts: List of reference text strings (optional)
            
        Returns:
            Comprehensive metrics dictionary
        """
        # Start with batch metrics
        metrics = self.compute_batch_metrics(predictions, labels, input_ids)
        
        # Add text-based metrics if available
        if predicted_texts and reference_texts:
            # BLEU scores
            bleu_metrics = self.compute_bleu_score(predicted_texts, reference_texts)
            metrics.update(bleu_metrics)
            
            # ROUGE scores
            rouge_metrics = self.compute_rouge_score(predicted_texts, reference_texts)
            metrics.update(rouge_metrics)
            
            # Medical-specific metrics
            metrics['medical_terminology_accuracy'] = self.compute_medical_terminology_accuracy(
                predicted_texts, reference_texts
            )
            metrics['semantic_similarity'] = self.compute_semantic_similarity(
                predicted_texts, reference_texts
            )
        
        return metrics

class FederatedMetricsAggregator:
    """
    Aggregates metrics across federated clients
    """
    
    def __init__(self):
        """Initialize metrics aggregator"""
        self.client_metrics = defaultdict(list)
        self.round_metrics = []
        
    def add_client_metrics(self, client_id: int, metrics: Dict[str, float]):
        """
        Add metrics from a client
        
        Args:
            client_id: Client identifier
            metrics: Dictionary of metrics
        """
        for metric_name, value in metrics.items():
            self.client_metrics[f"client_{client_id}_{metric_name}"].append(value)
    
    def add_server_metrics(self, metrics: Dict[str, float]):
        """
        Add metrics from server
        
        Args:
            metrics: Dictionary of server metrics
        """
        for metric_name, value in metrics.items():
            self.client_metrics[f"server_{metric_name}"].append(value)
    
    def compute_federated_averages(self) -> Dict[str, float]:
        """
        Compute federated averages of metrics
        
        Returns:
            Dictionary of averaged metrics
        """
        federated_metrics = {}
        
        # Group metrics by type (ignoring client/server prefix)
        metric_groups = defaultdict(list)
        
        for full_metric_name, values in self.client_metrics.items():
            if not values:
                continue
                
            # Extract base metric name
            if full_metric_name.startswith('client_'):
                parts = full_metric_name.split('_', 2)
                if len(parts) >= 3:
                    base_metric = parts[2]
                else:
                    base_metric = full_metric_name
            elif full_metric_name.startswith('server_'):
                base_metric = full_metric_name[7:]  # Remove 'server_' prefix
            else:
                base_metric = full_metric_name
            
            metric_groups[base_metric].extend(values)
        
        # Compute averages
        for metric_name, all_values in metric_groups.items():
            if all_values:
                federated_metrics[f"fed_avg_{metric_name}"] = np.mean(all_values)
                federated_metrics[f"fed_std_{metric_name}"] = np.std(all_values)
                federated_metrics[f"fed_min_{metric_name}"] = np.min(all_values)
                federated_metrics[f"fed_max_{metric_name}"] = np.max(all_values)
        
        return federated_metrics
    
    def compute_client_fairness_metrics(self) -> Dict[str, float]:
        """
        Compute fairness metrics across clients
        
        Returns:
            Dictionary of fairness metrics
        """
        fairness_metrics = {}
        
        # Extract client-specific metrics
        client_specific_metrics = defaultdict(dict)
        
        for full_metric_name, values in self.client_metrics.items():
            if full_metric_name.startswith('client_') and values:
                parts = full_metric_name.split('_', 2)
                if len(parts) >= 3:
                    client_id = int(parts[1])
                    metric_name = parts[2]
                    client_specific_metrics[client_id][metric_name] = np.mean(values)
        
        # Compute fairness metrics for each metric type
        metric_types = set()
        for client_metrics in client_specific_metrics.values():
            metric_types.update(client_metrics.keys())
        
        for metric_type in metric_types:
            client_values = []
            for client_metrics in client_specific_metrics.values():
                if metric_type in client_metrics:
                    client_values.append(client_metrics[metric_type])
            
            if len(client_values) > 1:
                # Coefficient of variation (CV) as fairness measure
                mean_val = np.mean(client_values)
                std_val = np.std(client_values)
                cv = std_val / (mean_val + 1e-8)  # Add small epsilon to avoid division by zero
                
                fairness_metrics[f"fairness_cv_{metric_type}"] = cv
                fairness_metrics[f"fairness_range_{metric_type}"] = np.max(client_values) - np.min(client_values)
        
        return fairness_metrics
    
    def get_training_summary(self) -> Dict[str, any]:
        """Get comprehensive training summary"""
        summary = {
            'federated_averages': self.compute_federated_averages(),
            'fairness_metrics': self.compute_client_fairness_metrics(),
            'total_metrics_collected': len(self.client_metrics),
            'rounds_completed': len(self.round_metrics)
        }
        
        return summary

# Test function
def test_medical_qa_metrics():
    """Test medical QA metrics"""
    print("Testing Medical QA Metrics...")
    
    # Create sample data
    batch_size, seq_len, vocab_size = 4, 10, 1000
    predictions = torch.softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)
    labels = torch.randint(1, vocab_size, (batch_size, seq_len))  # Avoid 0 (padding)
    
    # Initialize metrics
    metrics_calculator = MedicalQAMetrics()
    
    # Test batch metrics
    batch_metrics = metrics_calculator.compute_batch_metrics(predictions, labels)
    print("Batch Metrics:")
    for key, value in batch_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test text-based metrics
    predicted_texts = [
        "The patient has hypertension and requires medication.",
        "Diabetes symptoms include increased thirst and fatigue.",
        "Treatment involves insulin therapy and diet modification.",
        "The diagnosis is pneumonia with bacterial infection."
    ]
    
    reference_texts = [
        "Patient diagnosed with high blood pressure needs treatment.",
        "Diabetes causes excessive thirst and tiredness in patients.",
        "Insulin and dietary changes are recommended for treatment.",
        "Bacterial pneumonia diagnosis confirmed through chest X-ray."
    ]
    
    # Comprehensive metrics
    comprehensive_metrics = metrics_calculator.compute_comprehensive_metrics(
        predictions, labels, 
        predicted_texts=predicted_texts,
        reference_texts=reference_texts
    )
    
    print("\nComprehensive Metrics:")
    for key, value in comprehensive_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test federated aggregator
    print("\nTesting Federated Metrics Aggregator...")
    aggregator = FederatedMetricsAggregator()
    
    # Add metrics from multiple clients
    for client_id in range(3):
        client_metrics = {
            'accuracy': 0.8 + np.random.normal(0, 0.1),
            'f1': 0.75 + np.random.normal(0, 0.1),
            'bleu1': 0.6 + np.random.normal(0, 0.1)
        }
        aggregator.add_client_metrics(client_id, client_metrics)
    
    # Add server metrics
    server_metrics = {'global_accuracy': 0.85, 'global_loss': 0.3}
    aggregator.add_server_metrics(server_metrics)
    
    # Get federated averages
    fed_averages = aggregator.compute_federated_averages()
    fairness_metrics = aggregator.compute_client_fairness_metrics()
    
    print("Federated Averages:")
    for key, value in fed_averages.items():
        print(f"  {key}: {value:.4f}")
    
    print("Fairness Metrics:")
    for key, value in fairness_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nAll metrics tests completed successfully!")

if __name__ == "__main__":
    test_medical_qa_metrics()
