#!/usr/bin/env python3
"""
Dataset Setup Script for Real Medical QA Datasets
Downloads and prepares real medical datasets from various sources
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.medical_qa_loader import MedicalQADataLoader

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dataset_setup.log')
        ]
    )
    return logging.getLogger(__name__)

def install_requirements():
    """Install additional requirements for real datasets"""
    logger = logging.getLogger(__name__)
    logger.info("Installing additional requirements for real medical datasets...")
    
    additional_packages = [
        'datasets>=2.12.0',
        'requests>=2.31.0', 
        'lxml>=4.9.0',
        'huggingface-hub>=0.16.0',
        'nltk>=3.8.0',
        'rouge-score>=0.1.2',
        'sacrebleu>=2.3.1'
    ]
    
    for package in additional_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")

def download_nltk_data():
    """Download required NLTK data"""
    logger = logging.getLogger(__name__)
    logger.info("Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")

def setup_huggingface_cache():
    """Setup Hugging Face cache directory"""
    logger = logging.getLogger(__name__)
    
    cache_dir = Path.home() / '.cache' / 'huggingface'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variable
    os.environ['HF_HOME'] = str(cache_dir)
    logger.info(f"Hugging Face cache directory set to: {cache_dir}")

def test_dataset_loading(dataset_names: list, max_samples: int = 100):
    """Test loading of medical datasets"""
    logger = logging.getLogger(__name__)
    
    # Create data loader
    loader = MedicalQADataLoader(cache_dir="./cache")
    
    results = {}
    
    for dataset_name in dataset_names:
        logger.info(f"Testing {dataset_name} dataset loading...")
        
        try:
            if dataset_name == "combined":
                dataset = loader.create_dataset(
                    dataset_name="combined",
                    max_samples=max_samples * 5,  # More samples for combined
                    max_length=256
                )
            else:
                dataset = loader.create_dataset(
                    dataset_name=dataset_name,
                    max_samples=max_samples,
                    max_length=256
                )
            
            # Get statistics
            stats = loader.get_data_statistics(dataset)
            results[dataset_name] = {
                'success': True,
                'num_samples': stats['num_samples'],
                'avg_question_length': stats['avg_question_length'],
                'avg_answer_length': stats['avg_answer_length'],
                'samples_with_context': stats.get('samples_with_context', 0)
            }
            
            logger.info(f"✓ {dataset_name}: {stats['num_samples']} samples loaded")
            
        except Exception as e:
            logger.error(f"✗ Failed to load {dataset_name}: {e}")
            results[dataset_name] = {
                'success': False,
                'error': str(e)
            }
    
    return results

def print_dataset_summary(results: dict):
    """Print summary of dataset loading results"""
    print("\n" + "="*60)
    print("MEDICAL DATASET SETUP SUMMARY")
    print("="*60)
    
    total_samples = 0
    successful_datasets = 0
    
    for dataset_name, result in results.items():
        if result['success']:
            print(f"✓ {dataset_name.upper()}:")
            print(f"  - Samples: {result['num_samples']:,}")
            print(f"  - Avg Question Length: {result['avg_question_length']:.1f} words")
            print(f"  - Avg Answer Length: {result['avg_answer_length']:.1f} words")
            if 'samples_with_context' in result:
                print(f"  - Samples with Context: {result['samples_with_context']:,}")
            print()
            
            total_samples += result['num_samples']
            successful_datasets += 1
        else:
            print(f"✗ {dataset_name.upper()}: FAILED")
            print(f"  - Error: {result['error']}")
            print()
    
    print(f"TOTAL SUCCESSFUL DATASETS: {successful_datasets}")
    print(f"TOTAL SAMPLES AVAILABLE: {total_samples:,}")
    print("="*60)
    
    if successful_datasets > 0:
        print("\n🎉 Dataset setup completed successfully!")
        print("You can now run federated training with real medical data:")
        print("  python scripts/run_training.py --config config/config.yaml")
    else:
        print("\n⚠️  No datasets were loaded successfully.")
        print("Please check the error messages above and ensure you have:")
        print("  - Internet connection for downloading datasets")
        print("  - Sufficient disk space")
        print("  - Required dependencies installed")

def create_dataset_info_file(results: dict):
    """Create a file with dataset information"""
    info_file = Path("datasets_info.yaml")
    
    import yaml
    
    dataset_info = {
        'setup_timestamp': str(Path(__file__).stat().st_mtime),
        'datasets': {}
    }
    
    for dataset_name, result in results.items():
        if result['success']:
            dataset_info['datasets'][dataset_name] = {
                'num_samples': result['num_samples'],
                'avg_question_length': result['avg_question_length'],
                'avg_answer_length': result['avg_answer_length'],
                'samples_with_context': result.get('samples_with_context', 0),
                'status': 'available'
            }
        else:
            dataset_info['datasets'][dataset_name] = {
                'status': 'failed',
                'error': result['error']
            }
    
    with open(info_file, 'w') as f:
        yaml.dump(dataset_info, f, default_flow_style=False)
    
    print(f"\nDataset information saved to: {info_file}")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Setup Real Medical QA Datasets')
    parser.add_argument('--datasets', nargs='+', 
                       default=['pubmed_qa', 'medmcqa', 'head_qa', 'medical_meadow', 'combined'],
                       help='Datasets to setup and test')
    parser.add_argument('--max-samples', type=int, default=500,
                       help='Maximum samples to test load from each dataset')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install additional dependencies')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--cache-dir', default='./cache',
                       help='Directory to cache downloaded datasets')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("Starting Medical QA Dataset Setup...")
    logger.info(f"Datasets to setup: {args.datasets}")
    logger.info(f"Max samples per dataset: {args.max_samples}")
    
    # Create cache directory
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cache directory: {cache_dir.absolute()}")
    
    try:
        # Install dependencies if requested
        if args.install_deps:
            install_requirements()
        
        # Setup Hugging Face cache
        setup_huggingface_cache()
        
        # Download NLTK data
        download_nltk_data()
        
        # Test dataset loading
        logger.info("Testing dataset loading...")
        results = test_dataset_loading(args.datasets, args.max_samples)
        
        # Print summary
        print_dataset_summary(results)
        
        # Create info file
        create_dataset_info_file(results)
        
        # Check if any datasets were successful
        successful_count = sum(1 for r in results.values() if r['success'])
        
        if successful_count == 0:
            logger.error("No datasets were loaded successfully!")
            sys.exit(1)
        else:
            logger.info(f"Successfully setup {successful_count}/{len(results)} datasets")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
