import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

class TrainingAnalyzer:
    """Analyze and visualize training results"""
    
    def __init__(self, history_path='./logs/training_history.json'):
        with open(history_path, 'r') as f:
            self.history = json.load(f)
        
        self.organs = ['prostate', 'bladder', 'rectum']
        self.output_dir = Path('./analysis')
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_loss_curves(self):
        """Plot training and validation loss"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved loss curves to {self.output_dir / 'loss_curves.png'}")
    
    def plot_dice_scores(self):
        """Plot Dice scores for all organs"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = range(1, len(self.history['train_dice']['mean']) + 1)
        
        # Plot for each organ
        for idx, organ in enumerate(self.organs):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            ax.plot(epochs, self.history['train_dice'][organ], 
                   'b-', label=f'Train {organ.capitalize()}', linewidth=2, alpha=0.7)
            ax.plot(epochs, self.history['val_dice'][organ], 
                   'r-', label=f'Val {organ.capitalize()}', linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Dice Score', fontsize=11)
            ax.set_title(f'{organ.capitalize()} Dice Score', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        # Plot mean Dice
        ax = axes[1, 1]
        ax.plot(epochs, self.history['train_dice']['mean'], 
               'b-', label='Train Mean', linewidth=2, alpha=0.7)
        ax.plot(epochs, self.history['val_dice']['mean'], 
               'r-', label='Val Mean', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Dice Score', fontsize=11)
        ax.set_title('Mean Dice Score', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dice_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved Dice scores to {self.output_dir / 'dice_scores.png'}")
    
    def plot_hausdorff_distances(self):
        """Plot Hausdorff distances"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = range(1, len(self.history['val_hausdorff']['mean']) + 1)
        
        colors = ['blue', 'orange', 'green']
        
        # Plot for each organ
        for idx, organ in enumerate(self.organs):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            ax.plot(epochs, self.history['val_hausdorff'][organ], 
                   color=colors[idx], linewidth=2, marker='o', markersize=3)
            
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Hausdorff Distance (mm)', fontsize=11)
            ax.set_title(f'{organ.capitalize()} Hausdorff Distance', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Plot mean HD
        ax = axes[1, 1]
        ax.plot(epochs, self.history['val_hausdorff']['mean'], 
               'r-', linewidth=2, marker='o', markersize=3)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Hausdorff Distance (mm)', fontsize=11)
        ax.set_title('Mean Hausdorff Distance', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hausdorff_distances.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved Hausdorff distances to {self.output_dir / 'hausdorff_distances.png'}")
    
    def plot_learning_rate(self):
        """Plot learning rate schedule"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        epochs = range(1, len(self.history['learning_rate']) + 1)
        
        ax.plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_rate.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved learning rate to {self.output_dir / 'learning_rate.png'}")
    
    def plot_resource_usage(self):
        """Plot CPU/GPU usage over training"""
        if not self.history['resource_usage']:
            print("⚠ No resource usage data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['resource_usage']) + 1)
        
        # Extract metrics
        cpu_usage = [r['cpu_percent'] for r in self.history['resource_usage']]
        ram_usage = [r['ram_percent'] for r in self.history['resource_usage']]
        
        # CPU Usage
        axes[0, 0].plot(epochs, cpu_usage, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=11)
        axes[0, 0].set_ylabel('CPU Usage (%)', fontsize=11)
        axes[0, 0].set_title('CPU Usage', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 100])
        
        # RAM Usage
        axes[0, 1].plot(epochs, ram_usage, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=11)
        axes[0, 1].set_ylabel('RAM Usage (%)', fontsize=11)
        axes[0, 1].set_title('RAM Usage', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 100])
        
        # GPU Usage (if available)
        gpu_util = [r.get('gpu_utilization', 0) for r in self.history['resource_usage']]
        gpu_mem = [r.get('gpu_memory_percent', 0) for r in self.history['resource_usage']]
        
        if any(gpu_util):
            axes[1, 0].plot(epochs, gpu_util, 'r-', linewidth=2)
            axes[1, 0].set_xlabel('Epoch', fontsize=11)
            axes[1, 0].set_ylabel('GPU Utilization (%)', fontsize=11)
            axes[1, 0].set_title('GPU Utilization', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim([0, 100])
            
            axes[1, 1].plot(epochs, gpu_mem, 'm-', linewidth=2)
            axes[1, 1].set_xlabel('Epoch', fontsize=11)
            axes[1, 1].set_ylabel('GPU Memory (%)', fontsize=11)
            axes[1, 1].set_title('GPU Memory Usage', fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 100])
        else:
            axes[1, 0].text(0.5, 0.5, 'No GPU Data', ha='center', va='center', fontsize=14)
            axes[1, 1].text(0.5, 0.5, 'No GPU Data', ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'resource_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved resource usage to {self.output_dir / 'resource_usage.png'}")
    
    def plot_combined_metrics(self):
        """Plot all key metrics in one figure"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2, alpha=0.7)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Dice scores
        ax2 = fig.add_subplot(gs[1, 0])
        for organ in self.organs:
            ax2.plot(epochs, self.history['val_dice'][organ], linewidth=2, label=organ.capitalize())
        ax2.set_ylabel('Dice Score', fontsize=11)
        ax2.set_title('Validation Dice Scores', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # Mean Dice
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(epochs, self.history['val_dice']['mean'], 'g-', linewidth=2.5)
        ax3.set_ylabel('Mean Dice Score', fontsize=11)
        ax3.set_title('Mean Validation Dice', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
        
        # Hausdorff Distance
        ax4 = fig.add_subplot(gs[1, 2])
        for organ in self.organs:
            ax4.plot(epochs, self.history['val_hausdorff'][organ], linewidth=2, label=organ.capitalize())
        ax4.set_ylabel('HD (mm)', fontsize=11)
        ax4.set_title('Hausdorff Distance', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # Learning Rate
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(epochs, self.history['learning_rate'], 'purple', linewidth=2)
        ax5.set_xlabel('Epoch', fontsize=11)
        ax5.set_ylabel('Learning Rate', fontsize=11)
        ax5.set_title('Learning Rate', fontsize=12, fontweight='bold')
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3)
        
        # Resource usage
        if self.history['resource_usage']:
            cpu = [r['cpu_percent'] for r in self.history['resource_usage']]
            ram = [r['ram_percent'] for r in self.history['resource_usage']]
            
            ax6 = fig.add_subplot(gs[2, 1])
            ax6.plot(epochs, cpu, 'b-', linewidth=2, label='CPU')
            ax6.plot(epochs, ram, 'g-', linewidth=2, label='RAM')
            ax6.set_xlabel('Epoch', fontsize=11)
            ax6.set_ylabel('Usage (%)', fontsize=11)
            ax6.set_title('System Resources', fontsize=12, fontweight='bold')
            ax6.legend(fontsize=9)
            ax6.grid(True, alpha=0.3)
            ax6.set_ylim([0, 100])
            
            gpu_mem = [r.get('gpu_memory_percent', 0) for r in self.history['resource_usage']]
            if any(gpu_mem):
                ax7 = fig.add_subplot(gs[2, 2])
                ax7.plot(epochs, gpu_mem, 'r-', linewidth=2)
                ax7.set_xlabel('Epoch', fontsize=11)
                ax7.set_ylabel('GPU Memory (%)', fontsize=11)
                ax7.set_title('GPU Memory', fontsize=12, fontweight='bold')
                ax7.grid(True, alpha=0.3)
                ax7.set_ylim([0, 100])
        
        plt.savefig(self.output_dir / 'combined_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved combined metrics to {self.output_dir / 'combined_metrics.png'}")
    
    def generate_summary_report(self):
        """Generate text summary report"""
        report_path = self.output_dir / 'training_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TRAINING SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Best epoch
            best_epoch = np.argmax(self.history['val_dice']['mean']) + 1
            best_dice = max(self.history['val_dice']['mean'])
            
            f.write(f"Best Epoch: {best_epoch}\n")
            f.write(f"Best Validation Dice: {best_dice:.4f}\n\n")
            
            # Final metrics
            f.write("Final Epoch Metrics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Train Loss: {self.history['train_loss'][-1]:.4f}\n")
            f.write(f"  Val Loss: {self.history['val_loss'][-1]:.4f}\n\n")
            
            f.write("  Validation Dice Scores:\n")
            for organ in self.organs:
                dice = self.history['val_dice'][organ][-1]
                f.write(f"    {organ.capitalize()}: {dice:.4f}\n")
            f.write(f"    Mean: {self.history['val_dice']['mean'][-1]:.4f}\n\n")
            
            f.write("  Hausdorff Distances (mm):\n")
            for organ in self.organs:
                hd = self.history['val_hausdorff'][organ][-1]
                f.write(f"    {organ.capitalize()}: {hd:.2f}\n")
            f.write(f"    Mean: {self.history['val_hausdorff']['mean'][-1]:.2f}\n\n")
            
            # Best metrics
            f.write("Best Validation Metrics:\n")
            f.write("-" * 40 + "\n")
            for organ in self.organs:
                best_dice_organ = max(self.history['val_dice'][organ])
                best_hd_organ = min(self.history['val_hausdorff'][organ])
                f.write(f"  {organ.capitalize()}:\n")
                f.write(f"    Best Dice: {best_dice_organ:.4f}\n")
                f.write(f"    Best HD: {best_hd_organ:.2f} mm\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"✓ Saved summary report to {report_path}")
    
    def analyze_all(self):
        """Run all analyses"""
        print("\n" + "="*80)
        print("GENERATING ANALYSIS PLOTS AND REPORTS")
        print("="*80 + "\n")
        
        self.plot_loss_curves()
        self.plot_dice_scores()
        self.plot_hausdorff_distances()
        self.plot_learning_rate()
        self.plot_resource_usage()
        self.plot_combined_metrics()
        self.generate_summary_report()
        
        print("\n" + "="*80)
        print(f"✓ All analyses complete! Results saved to: {self.output_dir}")
        print("="*80 + "\n")

def main():
    analyzer = TrainingAnalyzer('./logs/training_history.json')
    analyzer.analyze_all()

if __name__ == "__main__":
    main()

