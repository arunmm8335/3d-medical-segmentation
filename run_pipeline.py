#!/usr/bin/env python3
"""
Complete pipeline for medical image segmentation
Runs preprocessing, training, evaluation, and analysis
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            'processed_data',
            'checkpoints',
            'logs',
            'evaluation_results',
            'analysis'
        ]
        for d in dirs:
            Path(d).mkdir(exist_ok=True)
        print("✓ Directories created/verified\n")
    
    def run_preprocessing(self):
        """Run data preprocessing"""
        print("="*80)
        print("STEP 1: DATA PREPROCESSING")
        print("="*80 + "\n")
        
        if Path('preprocessing.py').exists():
            print("Running preprocessing.py...")
            subprocess.run([sys.executable, 'preprocessing.py'], check=True)
            print("\n✓ Preprocessing complete\n")
        else:
            print("⚠ preprocessing.py not found. Skipping...")
            print("  Make sure data is already preprocessed in ./processed_data/\n")
    
    def run_training(self):
        """Run model training"""
        print("="*80)
        print("STEP 2: MODEL TRAINING")
        print("="*80 + "\n")
        
        print("Starting training...")
        subprocess.run([sys.executable, 'train.py'], check=True)
        print("\n✓ Training complete\n")
    
    def run_evaluation(self):
        """Run model evaluation"""
        print("="*80)
        print("STEP 3: MODEL EVALUATION")
        print("="*80 + "\n")
        
        model_path = Path('checkpoints/best_model.pth')
        if not model_path.exists():
            print(f"✗ Model not found at {model_path}")
            print("  Please train the model first or specify correct path\n")
            return
        
        print("Evaluating model...")
        subprocess.run([
            sys.executable, 'evaluate.py',
            '--model', str(model_path),
            '--data', './processed_data',
            '--output', './evaluation_results',
            '--device', self.config['device']
        ], check=True)
        print("\n✓ Evaluation complete\n")
    
    def run_analysis(self):
        """Run analysis and visualization"""
        print("="*80)
        print("STEP 4: ANALYSIS AND VISUALIZATION")
        print("="*80 + "\n")
        
        history_path = Path('logs/training_history.json')
        if not history_path.exists():
            print(f"✗ Training history not found at {history_path}")
            print("  Please train the model first\n")
            return
        
        print("Generating analysis plots...")
        subprocess.run([sys.executable, 'visualize_analysis.py'], check=True)
        print("\n✓ Analysis complete\n")
    
    def print_summary(self):
        """Print pipeline summary"""
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80 + "\n")
        
        print("📁 Output Locations:")
        print("  - Preprocessed data: ./processed_data/")
        print("  - Model checkpoints: ./checkpoints/")
        print("  - Training logs: ./logs/")
        print("  - Evaluation results: ./evaluation_results/")
        print("  - Analysis plots: ./analysis/")
        
        print("\n📊 Key Files:")
        
        files_to_check = [
            ('checkpoints/best_model.pth', 'Best model checkpoint'),
            ('logs/training_history.json', 'Training history'),
            ('evaluation_results/evaluation_report.txt', 'Evaluation report'),
            ('analysis/combined_metrics.png', 'Combined metrics plot'),
            ('analysis/training_summary.txt', 'Training summary')
        ]
        
        for filepath, description in files_to_check:
            if Path(filepath).exists():
                print(f"  ✓ {description}: {filepath}")
            else:
                print(f"  ✗ {description}: {filepath} (not found)")
        
        print("\n" + "="*80 + "\n")
    
    def run_all(self):
        """Run complete pipeline"""
        print("\n" + "="*80)
        print("MEDICAL IMAGE SEGMENTATION PIPELINE")
        print("="*80 + "\n")
        
        try:
            if self.config['run_preprocessing']:
                self.run_preprocessing()
            
            if self.config['run_training']:
                self.run_training()
            
            if self.config['run_evaluation']:
                self.run_evaluation()
            
            if self.config['run_analysis']:
                self.run_analysis()
            
            self.print_summary()
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Pipeline failed at: {e.cmd}")
            print(f"  Error code: {e.returncode}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n\n⚠ Pipeline interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Run complete medical image segmentation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_pipeline.py --all
  
  # Run only training
  python run_pipeline.py --train
  
  # Run evaluation and analysis
  python run_pipeline.py --eval --analyze
  
  # Run everything except preprocessing
  python run_pipeline.py --all --no-preprocess
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline (preprocess, train, eval, analyze)')
    parser.add_argument('--preprocess', action='store_true',
                       help='Run data preprocessing')
    parser.add_argument('--train', action='store_true',
                       help='Run model training')
    parser.add_argument('--eval', action='store_true',
                       help='Run model evaluation')
    parser.add_argument('--analyze', action='store_true',
                       help='Run analysis and visualization')
    parser.add_argument('--no-preprocess', action='store_true',
                       help='Skip preprocessing (use with --all)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Determine what to run
    if args.all:
        config = {
            'run_preprocessing': not args.no_preprocess,
            'run_training': True,
            'run_evaluation': True,
            'run_analysis': True,
            'device': args.device
        }
    else:
        # If nothing specified, show help
        if not (args.preprocess or args.train or args.eval or args.analyze):
            parser.print_help()
            sys.exit(0)
        
        config = {
            'run_preprocessing': args.preprocess,
            'run_training': args.train,
            'run_evaluation': args.eval,
            'run_analysis': args.analyze,
            'device': args.device
        }
    
    # Run pipeline
    pipeline = Pipeline(config)
    pipeline.run_all()

if __name__ == "__main__":
    main()
