"""
Medical QA Dataset Loader - Real Datasets Only
Loads and preprocesses medical question-answering datasets from reputable sources:
- PubMedQA: Biomedical research questions from PubMed abstracts
- MedMCQA: Medical entrance exam questions (AIIMS & NEET)
- HEAD-QA: Spanish healthcare system exam questions
- MedQuAD: Medical questions from NIH and other authoritative sources
- HealthSearchQA: Real medical questions from search logs
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset as HFDataset
import pandas as pd
from typing import Dict, List, Tuple, Optional
import numpy as np
import json
import requests
import zipfile
import os
from pathlib import Path
import logging

class MedicalQADataset(Dataset):
    """Medical Question Answering Dataset"""
    
    def __init__(self, 
                 questions: List[str], 
                 answers: List[str],
                 contexts: Optional[List[str]] = None,
                 tokenizer: AutoTokenizer = None,
                 max_length: int = 512,
                 task_type: str = "qa"):
        """
        Args:
            questions: List of medical questions
            answers: List of corresponding answers
            contexts: Optional list of contexts/abstracts
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            task_type: Type of task ('qa', 'classification', etc.)
        """
        self.questions = questions
        self.answers = answers
        self.contexts = contexts or [None] * len(questions)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        
        assert len(questions) == len(answers), "Questions and answers must have same length"
        assert len(questions) == len(self.contexts), "Questions and contexts must have same length"
        
    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        question = str(self.questions[idx])
        answer = str(self.answers[idx])
        context = str(self.contexts[idx]) if self.contexts[idx] is not None else ""
        
        # Create input text by combining context (if available), question and answer
        if self.task_type == "qa":
            if context and len(context.strip()) > 0:
                input_text = f"Context: {context}\nQuestion: {question}"
                target_text = f"Answer: {answer}"
            else:
                input_text = f"Question: {question}"
                target_text = f"Answer: {answer}"
        else:
            # For classification tasks
            if context and len(context.strip()) > 0:
                input_text = f"{context} {question}"
            else:
                input_text = question
            target_text = answer
        
        # Tokenize input and target if tokenizer is provided
        if self.tokenizer:
            # Tokenize input
            input_encoding = self.tokenizer(
                input_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Tokenize target
            target_encoding = self.tokenizer(
                target_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': input_encoding['input_ids'].squeeze(0),
                'attention_mask': input_encoding['attention_mask'].squeeze(0),
                'labels': target_encoding['input_ids'].squeeze(0),
                'target_attention_mask': target_encoding['attention_mask'].squeeze(0),
                'question': question,
                'answer': answer,
                'context': context
            }
        else:
            # Return raw text if no tokenizer
            return {
                'input_text': input_text,
                'target_text': target_text,
                'question': question,
                'answer': answer,
                'context': context
            }

class MedicalQADataLoader:
    """Loads various real medical QA datasets from reputable sources"""
    
    def __init__(self, tokenizer_name: str = "microsoft/DialoGPT-medium", cache_dir: str = "./cache"):
        """
        Args:
            tokenizer_name: Name of the tokenizer to use
            cache_dir: Directory to cache downloaded datasets
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            logging.warning(f"Could not load tokenizer {tokenizer_name}: {e}")
            self.tokenizer = None
        
        # Add special tokens if needed
        if self.tokenizer and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_pubmed_qa(self, subset: str = "pqa_labeled", max_samples: Optional[int] = None) -> Dict[str, List]:
        """
        Load PubMedQA dataset - Real biomedical research questions
        
        Args:
            subset: Subset to load ('pqa_labeled', 'pqa_unlabeled', 'pqa_artificial')
            max_samples: Maximum number of samples to load
            
        Returns:
            Dictionary with questions, answers, and contexts
        """
        try:
            self.logger.info(f"Loading PubMedQA dataset (subset: {subset})")
            dataset = load_dataset("qiaojin/PubMedQA", subset, cache_dir=str(self.cache_dir))
            
            questions = []
            answers = []
            contexts = []
            
            sample_count = 0
            for split in dataset:
                for item in dataset[split]:
                    if max_samples and sample_count >= max_samples:
                        break
                        
                    question = item['question']
                    
                    # Handle different answer formats
                    if 'final_decision' in item and item['final_decision']:
                        answer = item['final_decision']
                    elif 'long_answer' in item and item['long_answer']:
                        answer = item['long_answer']
                    else:
                        continue  # Skip items without proper answers
                    
                    # Add context from abstract if available
                    context = ""
                    if 'context' in item and item['context']:
                        if isinstance(item['context'], dict) and 'contexts' in item['context']:
                            context = " ".join(item['context']['contexts'])
                        elif isinstance(item['context'], list):
                            context = " ".join(item['context'])
                        elif isinstance(item['context'], str):
                            context = item['context']
                    
                    questions.append(question)
                    answers.append(answer)
                    contexts.append(context)
                    sample_count += 1
            
            self.logger.info(f"Loaded {len(questions)} samples from PubMedQA")
            return {'questions': questions, 'answers': answers, 'contexts': contexts}
            
        except Exception as e:
            self.logger.error(f"Error loading PubMedQA: {e}")
            return {'questions': [], 'answers': [], 'contexts': []}
    
    def load_medmcqa(self, max_samples: Optional[int] = None) -> Dict[str, List]:
        """
        Load MedMCQA dataset - Medical entrance exam questions (AIIMS & NEET)
        
        Args:
            max_samples: Maximum number of samples to load
            
        Returns:
            Dictionary with questions, answers, and contexts
        """
        try:
            self.logger.info("Loading MedMCQA dataset")
            dataset = load_dataset("openlifescienceai/medmcqa", cache_dir=str(self.cache_dir))
            
            questions = []
            answers = []
            contexts = []
            
            sample_count = 0
            for split in ['train', 'validation', 'test']:
                if split not in dataset:
                    continue
                    
                for item in dataset[split]:
                    if max_samples and sample_count >= max_samples:
                        break
                    
                    question = item['question']
                    
                    # Get the correct answer
                    correct_answer_idx = item['cop'] - 1  # Convert to 0-based index
                    options = [item['opa'], item['opb'], item['opc'], item['opd']]
                    
                    if 0 <= correct_answer_idx < len(options):
                        answer = options[correct_answer_idx]
                    else:
                        continue  # Skip invalid items
                    
                    # Add explanation as context if available
                    context = item.get('exp', '')
                    
                    questions.append(question)
                    answers.append(answer)
                    contexts.append(context)
                    sample_count += 1
            
            self.logger.info(f"Loaded {len(questions)} samples from MedMCQA")
            return {'questions': questions, 'answers': answers, 'contexts': contexts}
            
        except Exception as e:
            self.logger.error(f"Error loading MedMCQA: {e}")
            return {'questions': [], 'answers': [], 'contexts': []}
    
    def load_head_qa(self, language: str = "en", max_samples: Optional[int] = None) -> Dict[str, List]:
        """
        Load HEAD-QA dataset - Spanish healthcare system exam questions
        
        Args:
            language: Language to load ('en' for English, 'es' for Spanish)
            max_samples: Maximum number of samples to load
            
        Returns:
            Dictionary with questions, answers, and contexts
        """
        try:
            self.logger.info(f"Loading HEAD-QA dataset (language: {language})")
            dataset = load_dataset("head_qa", language, cache_dir=str(self.cache_dir))
            
            questions = []
            answers = []
            contexts = []
            
            sample_count = 0
            for split in dataset:
                for item in dataset[split]:
                    if max_samples and sample_count >= max_samples:
                        break
                    
                    question = item['qtext']
                    
                    # Get the correct answer
                    correct_answer_id = int(item['ra'])
                    answer_text = None
                    
                    for answer_option in item['answers']:
                        if answer_option['aid'] == correct_answer_id:
                            answer_text = answer_option['atext']
                            break
                    
                    if not answer_text:
                        continue  # Skip if no correct answer found
                    
                    # Add category as context
                    context = f"Category: {item.get('category', 'Unknown')}"
                    
                    questions.append(question)
                    answers.append(answer_text)
                    contexts.append(context)
                    sample_count += 1
            
            self.logger.info(f"Loaded {len(questions)} samples from HEAD-QA")
            return {'questions': questions, 'answers': answers, 'contexts': contexts}
            
        except Exception as e:
            self.logger.error(f"Error loading HEAD-QA: {e}")
            return {'questions': [], 'answers': [], 'contexts': []}
    
    def load_medical_meadow_qa(self, max_samples: Optional[int] = None) -> Dict[str, List]:
        """
        Load Medical Meadow QA dataset - Medical flashcards and QA pairs
        
        Args:
            max_samples: Maximum number of samples to load
            
        Returns:
            Dictionary with questions, answers, and contexts
        """
        try:
            self.logger.info("Loading Medical Meadow QA dataset")
            dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards", cache_dir=str(self.cache_dir))
            
            questions = []
            answers = []
            contexts = []
            
            sample_count = 0
            for split in dataset:
                for item in dataset[split]:
                    if max_samples and sample_count >= max_samples:
                        break
                    
                    if 'input' in item and 'output' in item:
                        question = item['input']
                        answer = item['output']
                    elif 'question' in item and 'answer' in item:
                        question = item['question']
                        answer = item['answer']
                    else:
                        continue
                    
                    # Add instruction as context if available
                    context = item.get('instruction', '')
                    
                    questions.append(question)
                    answers.append(answer)
                    contexts.append(context)
                    sample_count += 1
            
            self.logger.info(f"Loaded {len(questions)} samples from Medical Meadow")
            return {'questions': questions, 'answers': answers, 'contexts': contexts}
            
        except Exception as e:
            self.logger.error(f"Error loading Medical Meadow: {e}")
            return {'questions': [], 'answers': [], 'contexts': []}
    
    def download_medquad_dataset(self) -> Dict[str, List]:
        """
        Download and load MedQuAD dataset from official source
        MedQuAD contains medical Q&A from authoritative sources like NIH
        
        Returns:
            Dictionary with questions, answers, and contexts
        """
        try:
            self.logger.info("Downloading MedQuAD dataset")
            
            # Official MedQuAD GitHub repository
            medquad_url = "https://github.com/abachaa/MedQuAD/archive/master.zip"
            cache_file = self.cache_dir / "medquad.zip"
            extract_dir = self.cache_dir / "MedQuAD-master"
            
            # Download if not cached
            if not cache_file.exists():
                response = requests.get(medquad_url)
                response.raise_for_status()
                with open(cache_file, 'wb') as f:
                    f.write(response.content)
                self.logger.info("Downloaded MedQuAD dataset")
            
            # Extract if not already extracted
            if not extract_dir.exists():
                with zipfile.ZipFile(cache_file, 'r') as zip_ref:
                    zip_ref.extractall(self.cache_dir)
                self.logger.info("Extracted MedQuAD dataset")
            
            # Parse the dataset
            questions = []
            answers = []
            contexts = []
            
            # Look for XML files in the MedQuAD directory structure
            for xml_dir in extract_dir.glob("*/"):
                if xml_dir.is_dir():
                    for xml_file in xml_dir.glob("*.xml"):
                        try:
                            import xml.etree.ElementTree as ET
                            tree = ET.parse(xml_file)
                            root = tree.getroot()
                            
                            # Parse QA pairs from XML
                            for qa_pair in root.findall('.//QAPair'):
                                question_elem = qa_pair.find('Question')
                                answer_elem = qa_pair.find('Answer')
                                
                                if question_elem is not None and answer_elem is not None:
                                    question = question_elem.text
                                    answer = answer_elem.text
                                    
                                    if question and answer:
                                        questions.append(question.strip())
                                        answers.append(answer.strip())
                                        contexts.append(f"Source: {xml_file.stem}")
                        except Exception as e:
                            self.logger.warning(f"Error parsing {xml_file}: {e}")
                            continue
            
            self.logger.info(f"Loaded {len(questions)} samples from MedQuAD")
            return {'questions': questions, 'answers': answers, 'contexts': contexts}
            
        except Exception as e:
            self.logger.error(f"Error loading MedQuAD: {e}")
            return {'questions': [], 'answers': [], 'contexts': []}
    
    def load_healthsearch_qa(self, max_samples: Optional[int] = None) -> Dict[str, List]:
        """
        Load HealthSearchQA dataset - Real medical questions from search logs
        
        Args:
            max_samples: Maximum number of samples to load
            
        Returns:
            Dictionary with questions, answers, and contexts
        """
        try:
            self.logger.info("Loading HealthSearchQA dataset")
            # Try to load from Hugging Face datasets
            dataset = load_dataset("arabic-nlp/HealthSearchQA", cache_dir=str(self.cache_dir))
            
            questions = []
            answers = []
            contexts = []
            
            sample_count = 0
            for split in dataset:
                for item in dataset[split]:
                    if max_samples and sample_count >= max_samples:
                        break
                    
                    question = item.get('question', '')
                    answer = item.get('answer', '')
                    
                    if question and answer:
                        context = item.get('url', '') or item.get('source', '')
                        
                        questions.append(question)
                        answers.append(answer)
                        contexts.append(context)
                        sample_count += 1
            
            self.logger.info(f"Loaded {len(questions)} samples from HealthSearchQA")
            return {'questions': questions, 'answers': answers, 'contexts': contexts}
            
        except Exception as e:
            self.logger.error(f"Error loading HealthSearchQA: {e}")
            return {'questions': [], 'answers': [], 'contexts': []}
    
    def create_combined_dataset(self, 
                              datasets: List[str] = None,
                              max_samples_per_dataset: int = 1000,
                              max_length: int = 512,
                              task_type: str = "qa") -> MedicalQADataset:
        """
        Create a combined MedicalQADataset from multiple real datasets
        
        Args:
            datasets: List of dataset names to combine. If None, uses all available
            max_samples_per_dataset: Maximum samples to load from each dataset
            max_length: Maximum sequence length
            task_type: Type of task
            
        Returns:
            Combined MedicalQADataset instance
        """
        if datasets is None:
            datasets = ['pubmed_qa', 'medmcqa', 'head_qa', 'medical_meadow', 'medquad']
        
        all_questions = []
        all_answers = []
        all_contexts = []
        
        for dataset_name in datasets:
            self.logger.info(f"Loading dataset: {dataset_name}")
            
            try:
                if dataset_name == "pubmed_qa":
                    data = self.load_pubmed_qa(max_samples=max_samples_per_dataset)
                elif dataset_name == "medmcqa":
                    data = self.load_medmcqa(max_samples=max_samples_per_dataset)
                elif dataset_name == "head_qa":
                    data = self.load_head_qa(max_samples=max_samples_per_dataset)
                elif dataset_name == "medical_meadow":
                    data = self.load_medical_meadow_qa(max_samples=max_samples_per_dataset)
                elif dataset_name == "medquad":
                    data = self.download_medquad_dataset()
                    if len(data['questions']) > max_samples_per_dataset:
                        # Randomly sample if too many samples
                        indices = np.random.choice(len(data['questions']), max_samples_per_dataset, replace=False)
                        data = {
                            'questions': [data['questions'][i] for i in indices],
                            'answers': [data['answers'][i] for i in indices],
                            'contexts': [data['contexts'][i] for i in indices]
                        }
                elif dataset_name == "healthsearch_qa":
                    data = self.load_healthsearch_qa(max_samples=max_samples_per_dataset)
                else:
                    self.logger.warning(f"Unknown dataset: {dataset_name}")
                    continue
                
                if data['questions']:
                    all_questions.extend(data['questions'])
                    all_answers.extend(data['answers'])
                    all_contexts.extend(data['contexts'])
                    self.logger.info(f"Added {len(data['questions'])} samples from {dataset_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load {dataset_name}: {e}")
                continue
        
        self.logger.info(f"Total combined dataset size: {len(all_questions)} samples")
        
        # Create and return dataset
        return MedicalQADataset(
            questions=all_questions,
            answers=all_answers,
            contexts=all_contexts,
            tokenizer=self.tokenizer,
            max_length=max_length,
            task_type=task_type
        )
    
    def create_dataset(self, 
                      dataset_name: str = "combined",
                      max_samples: int = 5000,
                      max_length: int = 512,
                      task_type: str = "qa") -> MedicalQADataset:
        """
        Create a MedicalQADataset from real medical data sources
        
        Args:
            dataset_name: Name of dataset to load or 'combined' for all
            max_samples: Maximum number of samples
            max_length: Maximum sequence length
            task_type: Type of task
            
        Returns:
            MedicalQADataset instance with real medical data
        """
        if dataset_name == "combined":
            return self.create_combined_dataset(
                max_samples_per_dataset=max_samples // 5,  # Distribute across 5 datasets
                max_length=max_length,
                task_type=task_type
            )
        else:
            # Load single dataset
            if dataset_name == "pubmed_qa":
                data = self.load_pubmed_qa(max_samples=max_samples)
            elif dataset_name == "medmcqa":
                data = self.load_medmcqa(max_samples=max_samples)
            elif dataset_name == "head_qa":
                data = self.load_head_qa(max_samples=max_samples)
            elif dataset_name == "medical_meadow":
                data = self.load_medical_meadow_qa(max_samples=max_samples)
            elif dataset_name == "medquad":
                data = self.download_medquad_dataset()
                if len(data['questions']) > max_samples:
                    indices = np.random.choice(len(data['questions']), max_samples, replace=False)
                    data = {
                        'questions': [data['questions'][i] for i in indices],
                        'answers': [data['answers'][i] for i in indices],
                        'contexts': [data['contexts'][i] for i in indices]
                    }
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            return MedicalQADataset(
                questions=data['questions'],
                answers=data['answers'],
                contexts=data['contexts'],
                tokenizer=self.tokenizer,
                max_length=max_length,
                task_type=task_type
            )
    
    def get_data_statistics(self, dataset: MedicalQADataset) -> Dict[str, float]:
        """Get statistics about the dataset"""
        if self.tokenizer:
            questions = [dataset[i]['question'] for i in range(len(dataset))]
            answers = [dataset[i]['answer'] for i in range(len(dataset))]
            contexts = [dataset[i]['context'] for i in range(len(dataset))]
        else:
            questions = dataset.questions
            answers = dataset.answers
            contexts = dataset.contexts
        
        # Length statistics
        question_lengths = [len(q.split()) for q in questions]
        answer_lengths = [len(a.split()) for a in answers]
        context_lengths = [len(c.split()) if c else 0 for c in contexts]
        
        stats = {
            'num_samples': len(dataset),
            'avg_question_length': np.mean(question_lengths),
            'avg_answer_length': np.mean(answer_lengths),
            'avg_context_length': np.mean(context_lengths),
            'max_question_length': np.max(question_lengths),
            'max_answer_length': np.max(answer_lengths),
            'max_context_length': np.max(context_lengths),
            'vocab_size': len(self.tokenizer.get_vocab()) if self.tokenizer else 0,
            'samples_with_context': sum(1 for c in contexts if c and len(c.strip()) > 0)
        }
        
        return stats

def create_federated_dataloaders(dataset: MedicalQADataset,
                                num_clients: int = 3,
                                batch_size: int = 8,
                                iid: bool = False,
                                alpha: float = 0.5) -> List[DataLoader]:
    """
    Split dataset among clients and create dataloaders
    
    Args:
        dataset: MedicalQADataset to split
        num_clients: Number of federated clients
        batch_size: Batch size for each client
        iid: Whether to use IID data distribution
        alpha: Dirichlet concentration parameter for non-IID split
        
    Returns:
        List of DataLoaders for each client
    """
    total_samples = len(dataset)
    
    if iid:
        # IID split - equal random splits
        samples_per_client = total_samples // num_clients
        indices = torch.randperm(total_samples).tolist()
        
        client_indices = []
        for i in range(num_clients):
            start = i * samples_per_client
            end = start + samples_per_client if i < num_clients - 1 else total_samples
            client_indices.append(indices[start:end])
    
    else:
        # Non-IID split using Dirichlet distribution
        min_samples_per_client = 10
        
        # Create Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Ensure minimum samples per client
        proportions = np.maximum(proportions, min_samples_per_client / total_samples)
        proportions = proportions / proportions.sum()
        
        # Calculate actual sample counts
        sample_counts = (proportions * total_samples).astype(int)
        sample_counts[-1] = total_samples - sample_counts[:-1].sum()  # Adjust last client
        
        # Random shuffle and split
        indices = torch.randperm(total_samples).tolist()
        client_indices = []
        start_idx = 0
        
        for count in sample_counts:
            client_indices.append(indices[start_idx:start_idx + count])
            start_idx += count
    
    # Create dataloaders for each client
    client_dataloaders = []
    for indices in client_indices:
        client_dataset = torch.utils.data.Subset(dataset, indices)
        client_dataloader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        client_dataloaders.append(client_dataloader)
    
    return client_dataloaders

def test_medical_qa_loader():
    """Test the medical QA data loader with real datasets"""
    print("Testing Medical QA Data Loader with Real Datasets...")
    
    # Initialize loader
    loader = MedicalQADataLoader()
    
    # Test individual datasets
    datasets_to_test = ['pubmed_qa', 'medmcqa', 'head_qa']
    
    for dataset_name in datasets_to_test:
        try:
            print(f"\nCreated {len(client_loaders)} client dataloaders")
            for i, client_loader in enumerate(client_loaders):
                print(f"  Client {i}: {len(client_loader)} batches, {len(client_loader.dataset)} samples")
            
            # Test batch loading
            if len(client_loaders) > 0 and len(client_loaders[0]) > 0:
                batch = next(iter(client_loaders[0]))
                print(f"\nBatch shapes from client 0:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}")
                    elif isinstance(value, list):
                        print(f"  {key}: list with {len(value)} items")
        else:
            print("  No data loaded for combined dataset")
            
    except Exception as e:
        print(f"  Error testing combined dataset: {e}")
    
    print("\nMedical QA Data Loader testing completed!")

if __name__ == "__main__":
    test_medical_qa_loader()nTesting {dataset_name}...")
            dataset = loader.create_dataset(dataset_name=dataset_name, max_samples=100, max_length=256)
            
            if len(dataset) > 0:
                # Get statistics
                stats = loader.get_data_statistics(dataset)
                print(f"Dataset Statistics for {dataset_name}:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                
                # Test single sample
                sample = dataset[0]
                print(f"\nSample data from {dataset_name}:")
                if isinstance(sample.get('input_ids'), torch.Tensor):
                    for key, value in sample.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: {value.shape}")
                        else:
                            print(f"  {key}: {type(value)} - {str(value)[:100]}...")
                else:
                    print(f"  Question: {sample['question'][:100]}...")
                    print(f"  Answer: {sample['answer'][:100]}...")
            else:
                print(f"  No data loaded for {dataset_name}")
                
        except Exception as e:
            print(f"  Error testing {dataset_name}: {e}")
    
    # Test combined dataset
    print(f"\nTesting combined dataset...")
    try:
        combined_dataset = loader.create_dataset(dataset_name="combined", max_samples=500, max_length=256)
        
        if len(combined_dataset) > 0:
            stats = loader.get_data_statistics(combined_dataset)
            print("Combined Dataset Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            # Create federated dataloaders
            client_loaders = create_federated_dataloaders(
                combined_dataset, num_clients=3, batch_size=4, iid=False
            )
            
            print(f"\
