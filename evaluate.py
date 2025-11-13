import torch
import numpy as np
from torch.utils.data import DataLoader
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from train import CTDataset # <--- MAKE SURE THIS IS HERE
import glob
from multi_decoder_model import MultiDecoderUNet3D
from metrics import compute_all_metrics
# ... the rest of the file
# ... the rest of the file

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model_path, data_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {model_path}...")
        # --- AFTER ---
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.model = MultiDecoderUNet3D(in_channels=1, num_classes_per_organ=1)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load test dataset
        # --- AFTER ---
        # Load validation dataset IDs
        print("Loading validation data...")
        all_files = [f.replace('.npz', '') for f in os.listdir(data_dir) if f.endswith('.npz')]
        np.random.seed(42) # Use the same seed as train.py to get the same split
        np.random.shuffle(all_files)

        split_idx = int(len(all_files) * 0.8) # 0.8 is the train_split from train.py
        val_ids = all_files[split_idx:] # Get only the validation IDs

        self.test_dataset = CTDataset(data_dir, patient_ids=val_ids) # <--- THIS LOADS ONLY 39 PATIENTS
        self.test_loader = DataLoader(
        self.test_dataset,
        batch_size=1, # Already 1, but good to confirm
        shuffle=False,
        num_workers=2
        )
        
        self.results = []
        self.organs = ['prostate', 'bladder', 'rectum']
        
        print(f"✓ Model loaded on {self.device}")
        print(f"✓ Test dataset: {len(self.test_dataset)} patients\n")
    
    def evaluate_patient(self, ct, mask, patient_id):
        """Evaluate a single patient"""
        with torch.no_grad():
            ct = ct.to(self.device)
            mask = mask.to(self.device)
            
            # Forward pass
            output = self.model(ct)
            pred_binary = (output > 0.5).float()
            
            # Compute all metrics
            metrics = compute_all_metrics(pred_binary, mask)
            metrics['patient_id'] = patient_id
            
            return metrics, output.cpu(), pred_binary.cpu()
    
    def evaluate_all(self):
        """Evaluate entire test set"""
        print("="*80)
        print("EVALUATING MODEL ON TEST SET")
        print("="*80 + "\n")
        
        all_predictions = []
        
        with torch.no_grad():
            for idx, (ct, mask) in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                patient_id = self.test_dataset.patient_files[idx].replace('.npz', '')
                
                metrics, output, pred_binary = self.evaluate_patient(ct, mask, patient_id)
                
                self.results.append(metrics)
                all_predictions.append({
                    'patient_id': patient_id,
                    'prediction': pred_binary.numpy(),
                    'ground_truth': mask.numpy(),
                    'output': output.numpy()
                })
        
        # Compute summary statistics
        self.compute_summary_stats()
        
        return all_predictions
    
    def compute_summary_stats(self):
        """Compute mean and std for all metrics"""
        self.summary = {}
        
        # Get all metric keys (excluding patient_id)
        metric_keys = [k for k in self.results[0].keys() if k != 'patient_id']
        
        for key in metric_keys:
            values = [r[key] for r in self.results]
            self.summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
    
    def save_results(self, output_dir='./evaluation_results'):
        """Save evaluation results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        with open(output_dir / 'detailed_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary statistics
        with open(output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(self.summary, f, indent=2)
        
        # Generate text report
        self.generate_report(output_dir)
        
        print(f"\n✓ Results saved to {output_dir}")
    
    def generate_report(self, output_dir):
        """Generate evaluation report"""
        report_path = output_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Number of test cases: {len(self.results)}\n")
            f.write(f"Device: {self.device}\n\n")
            
            # Dice scores
            f.write("DICE SCORES\n")
            f.write("-"*80 + "\n")
            for organ in self.organs:
                key = f'{organ}_dice'
                f.write(f"{organ.capitalize()}:\n")
                f.write(f"  Mean: {self.summary[key]['mean']:.4f} ± {self.summary[key]['std']:.4f}\n")
                f.write(f"  Median: {self.summary[key]['median']:.4f}\n")
                f.write(f"  Range: [{self.summary[key]['min']:.4f}, {self.summary[key]['max']:.4f}]\n\n")
            
            f.write(f"Mean (all organs):\n")
            f.write(f"  Mean: {self.summary['mean_dice']['mean']:.4f} ± {self.summary['mean_dice']['std']:.4f}\n")
            f.write(f"  Median: {self.summary['mean_dice']['median']:.4f}\n\n")
            
            # IoU scores
            f.write("IOU SCORES\n")
            f.write("-"*80 + "\n")
            for organ in self.organs:
                key = f'{organ}_iou'
                f.write(f"{organ.capitalize()}:\n")
                f.write(f"  Mean: {self.summary[key]['mean']:.4f} ± {self.summary[key]['std']:.4f}\n")
                f.write(f"  Median: {self.summary[key]['median']:.4f}\n\n")
            
            # Hausdorff distances
            f.write("HAUSDORFF DISTANCES (mm)\n")
            f.write("-"*80 + "\n")
            for organ in self.organs:
                key = f'{organ}_hd'
                f.write(f"{organ.capitalize()}:\n")
                f.write(f"  Mean: {self.summary[key]['mean']:.2f} ± {self.summary[key]['std']:.2f}\n")
                f.write(f"  Median: {self.summary[key]['median']:.2f}\n")
                f.write(f"  Range: [{self.summary[key]['min']:.2f}, {self.summary[key]['max']:.2f}]\n\n")
            
            # 95th percentile HD
            f.write("HAUSDORFF 95 DISTANCES (mm)\n")
            f.write("-"*80 + "\n")
            for organ in self.organs:
                key = f'{organ}_hd95'
                f.write(f"{organ.capitalize()}:\n")
                f.write(f"  Mean: {self.summary[key]['mean']:.2f} ± {self.summary[key]['std']:.2f}\n")
                f.write(f"  Median: {self.summary[key]['median']:.2f}\n\n")
            
            # Sensitivity, Specificity, Precision
            f.write("SENSITIVITY, SPECIFICITY, PRECISION\n")
            f.write("-"*80 + "\n")
            for organ in self.organs:
                f.write(f"{organ.capitalize()}:\n")
                f.write(f"  Sensitivity: {self.summary[f'{organ}_sensitivity']['mean']:.4f}\n")
                f.write(f"  Specificity: {self.summary[f'{organ}_specificity']['mean']:.4f}\n")
                f.write(f"  Precision: {self.summary[f'{organ}_precision']['mean']:.4f}\n")
                f.write(f"  F1 Score: {self.summary[f'{organ}_f1']['mean']:.4f}\n\n")
            
            f.write("="*80 + "\n")
    
    def plot_results(self, output_dir='./evaluation_results'):
        """Generate visualization plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Plot 1: Box plots for Dice scores
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, organ in enumerate(self.organs):
            dice_values = [r[f'{organ}_dice'] for r in self.results]
            axes[idx].boxplot(dice_values, labels=[organ.capitalize()])
            axes[idx].set_ylabel('Dice Score')
            axes[idx].set_title(f'{organ.capitalize()} Dice Distribution')
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dice_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Scatter plot - Dice vs Hausdorff
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, organ in enumerate(self.organs):
            dice_values = [r[f'{organ}_dice'] for r in self.results]
            hd_values = [r[f'{organ}_hd'] for r in self.results]
            
            axes[idx].scatter(dice_values, hd_values, alpha=0.6)
            axes[idx].set_xlabel('Dice Score')
            axes[idx].set_ylabel('Hausdorff Distance (mm)')
            axes[idx].set_title(f'{organ.capitalize()}')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dice_vs_hausdorff.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Heatmap of all metrics
        metrics_to_plot = ['dice', 'iou', 'sensitivity', 'specificity', 'precision', 'f1']
        data = []
        
        for organ in self.organs:
            row = [self.summary[f'{organ}_{m}']['mean'] for m in metrics_to_plot]
            data.append(row)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            data,
            annot=True,
            fmt='.3f',
            xticklabels=[m.upper() for m in metrics_to_plot],
            yticklabels=[o.capitalize() for o in self.organs],
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            ax=ax
        )
        ax.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plots saved to {output_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, default='./checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='./processed_data',
                       help='Path to test data directory')
    parser.add_argument('--output', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Evaluate model
    evaluator = ModelEvaluator(args.model, args.data, args.device)
    predictions = evaluator.evaluate_all()
    
    # Save results
    evaluator.save_results(args.output)
    evaluator.plot_results(args.output)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: {args.output}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
