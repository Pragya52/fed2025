"""
Comprehensive Medical QA Metrics and Validation
Enhanced version with medical-specific evaluation metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import math
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, matthews_corrcoef
)
import re
import logging
from transformers import AutoTokenizer

# Try to import optional dependencies with fallbacks
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False
    print("NLTK not available, some metrics will use fallback implementations")

try:
    from rouge_score import rouge_scorer
    rouge_available = True
except ImportError:
    rouge_available = False
    print("ROUGE not available, using fallback implementation")

class MedicalQAMetrics:
    """Enhanced medical QA metrics with comprehensive evaluation"""
    
    def __init__(self, tokenizer_name: Optional[str] = None):
        """
        Initialize medical QA metrics
        
        Args:
            tokenizer_name: Tokenizer for text processing
        """
        self.logger = logging.getLogger("MedicalQAMetrics")
        
        # Initialize tokenizer
        if tokenizer_name:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except:
                self.tokenizer = None
        else:
            self.tokenizer = None
        
        # Initialize ROUGE scorer if available
        if rouge_available:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
            )
        else:
            self.rouge_scorer = None
        
        # Initialize BLEU smoothing if available
        if nltk_available:
            self.smoothing_function = SmoothingFunction().method1
        else:
            self.smoothing_function = None
        
        # Medical vocabulary for domain-specific evaluation
        self.medical_terms = self._load_comprehensive_medical_vocabulary()
        
        # Medical specialties for category-wise evaluation
        self.medical_specialties = self._load_medical_specialties()
        
    def _load_comprehensive_medical_vocabulary(self) -> set:
        """Load comprehensive medical vocabulary"""
        medical_terms = {
            # Basic medical terms
            'diagnosis', 'treatment', 'symptom', 'disease', 'patient', 'therapy',
            'medication', 'prescription', 'dosage', 'side effect', 'adverse reaction',
            'clinical', 'pathology', 'anatomy', 'physiology', 'pharmacology',
            
            # Common conditions
            'hypertension', 'diabetes', 'cancer', 'infection', 'inflammation',
            'pneumonia', 'asthma', 'arthritis', 'migraine', 'depression',
            'anxiety', 'heart disease', 'stroke', 'kidney', 'liver',
            'tuberculosis', 'malaria', 'hiv', 'hepatitis', 'influenza',
            'covid', 'alzheimer', 'parkinson', 'epilepsy', 'copd',
            
            # Body systems
            'cardiovascular', 'respiratory', 'neurological', 'gastrointestinal',
            'musculoskeletal', 'endocrine', 'immune', 'reproductive',
            'urinary', 'dermatological', 'ophthalmological', 'otolaryngology',
            
            # Medical procedures
            'surgery', 'biopsy', 'endoscopy', 'imaging', 'ultrasound',
            'mri', 'ct scan', 'x-ray', 'ecg', 'ekg', 'blood test',
            'urine test', 'vaccination', 'immunization', 'chemotherapy',
            
            # Medications
            'antibiotic', 'antiviral', 'analgesic', 'anti-inflammatory',
            'steroid', 'insulin', 'beta-blocker', 'ace inhibitor',
            'diuretic', 'anticoagulant', 'statin', 'proton pump inhibitor',
            
            # Medical measurements
            'blood pressure', 'heart rate', 'temperature', 'glucose',
            'cholesterol', 'hemoglobin', 'white blood cell', 'platelet',
            'creatinine', 'bilirubin', 'albumin', 'sodium', 'potassium'
        }
        
        return medical_terms
    
    def _load_medical_specialties(self) -> Dict[str, List[str]]:
        """Load medical specialties for category-wise evaluation"""
        return {
            'cardiology': ['heart', 'cardiac', 'cardiovascular', 'arrhythmia', 'myocardial'],
            'neurology': ['brain', 'neurological', 'seizure', 'stroke', 'dementia'],
            'endocrinology': ['diabetes', 'insulin', 'thyroid', 'hormone', 'glucose'],
            'pulmonology': ['lung', 'respiratory', 'asthma', 'pneumonia', 'copd'],
            'gastroenterology': ['stomach', 'liver', 'intestinal', 'digestive', 'hepatitis'],
            'nephrology': ['kidney', 'renal', 'dialysis', 'creatinine', 'urinary'],
            'oncology': ['cancer', 'tumor', 'malignant', 'chemotherapy', 'radiation'],
            'infectious_disease': ['infection', 'antibiotic', 'viral', 'bacterial', 'fungal']
        }
    
    def compute_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor, 
                        ignore_index: int = -100) -> float:
        """Compute accuracy with proper handling of ignored tokens"""
        if predictions.dim() == 3:  # [batch, seq, vocab]
            pred_classes = torch.argmax(predictions, dim=-1)
        else:
            pred_classes = predictions
        
        # Flatten
        pred_flat = pred_classes.view(-1)
        labels_flat = labels.view(-1)
        
        # Create mask for valid tokens
        mask = labels_flat != ignore_index
        
        if mask.sum() > 0:
            correct = (pred_flat[mask] == labels_flat[mask]).float()
            accuracy = correct.mean().item()
        else:
            accuracy = 0.0
        
        return accuracy
    
    def compute_top_k_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor, 
                             k: int = 5, ignore_index: int = -100) -> float:
        """Compute top-k accuracy"""
        if predictions.dim() != 3:
            return 0.0
        
        # Get top-k predictions
        _, top_k_preds = torch.topk(predictions, k, dim=-1)
        
        # Expand labels to match top-k shape
        labels_expanded = labels.unsqueeze(-1).expand_as(top_k_preds)
        
        # Check if true label is in top-k
        correct = (top_k_preds == labels_expanded).any(dim=-1)
        
        # Apply mask for valid tokens
        mask = labels != ignore_index
        
        if mask.sum() > 0:
            top_k_acc = correct[mask].float().mean().item()
        else:
            top_k_acc = 0.0
        
        return top_k_acc
    
    def compute_perplexity(self, predictions: torch.Tensor, labels: torch.Tensor, 
                          ignore_index: int = -100) -> float:
        """Compute perplexity with numerical stability"""
        if predictions.dim() != 3:
            return float('inf')
        
        # Flatten for loss computation
        predictions_flat = predictions.view(-1, predictions.size(-1))
        labels_flat = labels.view(-1)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            predictions_flat, 
            labels_flat, 
            ignore_index=ignore_index,
            reduction='mean'
        )
        
        # Compute perplexity with clipping to prevent overflow
        perplexity = torch.exp(torch.clamp(loss, max=10.0)).item()
        
        return perplexity
    
    def compute_f1_metrics(self, predictions: torch.Tensor, labels: torch.Tensor,
                          ignore_index: int = -100) -> Dict[str, float]:
        """Compute comprehensive F1 metrics"""
        if predictions.dim() == 3:
            pred_classes = torch.argmax(predictions, dim=-1)
        else:
            pred_classes = predictions
        
        # Flatten and filter valid tokens
        pred_flat = pred_classes.view(-1).cpu().numpy()
       
