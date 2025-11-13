import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import psutil
import GPUtil
from multi_decoder_model import MultiDecoderUNet3D

# Import metrics
from metrics import (
    compute_dice_score, 
    compute_hausdorff_distance,
    DiceLoss,
    CombinedLoss
)

class CTDataset(Dataset):
    """Dataset for loading preprocessed CT scans"""
    def __init__(self, data_dir, patient_ids=None):
        self.data_dir = data_dir
        if patient_ids is None:
            self.patient_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        else:
            self.patient_files = [f"{pid}.npz" for pid in patient_ids]
        
        print(f"Loaded {len(self.patient_files)} patients")
    
    def __len__(self):
        return len(self.patient_files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.patient_files[idx])
        data = np.load(file_path)
        
        ct = data['ct']  # Shape: (D, H, W, 1)
        mask = data['mask']  # Shape: (D, H, W, 3)
        
        # Convert to torch tensors and rearrange dimensions
        # From (D, H, W, C) to (C, D, H, W)
        ct = torch.from_numpy(ct).permute(3, 0, 1, 2).float()
        mask = torch.from_numpy(mask).permute(3, 0, 1, 2).float()
        
        return ct, mask

class ResourceMonitor:
    """Monitor CPU, GPU and memory usage"""
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        
    def get_metrics(self):
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / (1024**3),
            'ram_total_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics.update({
                        'gpu_utilization': gpu.load * 100,
                        'gpu_memory_used_mb': gpu.memoryUsed,
                        'gpu_memory_total_mb': gpu.memoryTotal,
                        'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'gpu_temperature': gpu.temperature
                    })
            except:
                pass
        
        return metrics

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = CombinedLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5
        )
        
        # Monitoring
        self.resource_monitor = ResourceMonitor()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': {'prostate': [], 'bladder': [], 'rectum': [], 'mean': []},
            'val_dice': {'prostate': [], 'bladder': [], 'rectum': [], 'mean': []},
            'val_hausdorff': {'prostate': [], 'bladder': [], 'rectum': [], 'mean': []},
            'learning_rate': [],
            'resource_usage': []
        }
        
        # Best model tracking
        self.best_val_dice = 0.0
        self.best_epoch = 0
        
        # Create output directories
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        dice_scores = {'prostate': [], 'bladder': [], 'rectum': []}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Train]")
        
        for batch_idx, (ct, mask) in enumerate(pbar):
            ct, mask = ct.to(self.device), mask.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(ct)
            
            # Compute loss
            loss = self.criterion(output, mask)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Compute metrics
            total_loss += loss.item()
            
            with torch.no_grad():
                pred_binary = (output > 0.5).float()
                for idx, organ in enumerate(['prostate', 'bladder', 'rectum']):
                    dice = compute_dice_score(
                        pred_binary[:, idx:idx+1], 
                        mask[:, idx:idx+1]
                    )
                    dice_scores[organ].append(dice)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'dice': np.mean([np.mean(v) for v in dice_scores.values()])
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_dice = {k: np.mean(v) for k, v in dice_scores.items()}
        avg_dice['mean'] = np.mean(list(avg_dice.values()))
        
        return avg_loss, avg_dice
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        dice_scores = {'prostate': [], 'bladder': [], 'rectum': []}
        hausdorff_distances = {'prostate': [], 'bladder': [], 'rectum': []}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Val]")
            
            for ct, mask in pbar:
                ct, mask = ct.to(self.device), mask.to(self.device)
                
                # Forward pass
                output = self.model(ct)
                
                # Compute loss
                loss = self.criterion(output, mask)
                total_loss += loss.item()
                
                # Compute metrics
                pred_binary = (output > 0.5).float()
                
                for idx, organ in enumerate(['prostate', 'bladder', 'rectum']):
                    pred_organ = pred_binary[:, idx:idx+1]
                    mask_organ = mask[:, idx:idx+1]
                    
                    dice = compute_dice_score(pred_organ, mask_organ)
                    dice_scores[organ].append(dice)
                    
                    # Compute Hausdorff distance
                    hd = compute_hausdorff_distance(
                        pred_organ.cpu().numpy(),
                        mask_organ.cpu().numpy()
                    )
                    hausdorff_distances[organ].append(hd)
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'dice': np.mean([np.mean(v) for v in dice_scores.values()])
                })
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_dice = {k: np.mean(v) for k, v in dice_scores.items()}
        avg_dice['mean'] = np.mean(list(avg_dice.values()))
        
        avg_hd = {k: np.mean(v) for k, v in hausdorff_distances.items()}
        avg_hd['mean'] = np.mean(list(avg_hd.values()))
        
        return avg_loss, avg_dice, avg_hd
    
    def train(self):
        print("\n" + "="*80)
        print(f"Starting Training: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            train_loss, train_dice = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_dice, val_hd = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step(val_dice['mean'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Monitor resources
            resource_metrics = self.resource_monitor.get_metrics()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            self.history['resource_usage'].append(resource_metrics)
            
            for organ in ['prostate', 'bladder', 'rectum', 'mean']:
                self.history['train_dice'][organ].append(train_dice[organ])
                self.history['val_dice'][organ].append(val_dice[organ])
                self.history['val_hausdorff'][organ].append(val_hd[organ])
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Dice: {train_dice['mean']:.4f} | Val Dice: {val_dice['mean']:.4f}")
            print(f"  Val HD (mean): {val_hd['mean']:.2f} mm")
            print(f"  LR: {current_lr:.6f}")
            print(f"  GPU Memory: {resource_metrics.get('gpu_memory_percent', 0):.1f}%")
            
            # Save best model
            if val_dice['mean'] > self.best_val_dice:
                self.best_val_dice = val_dice['mean']
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth', epoch, val_dice['mean'])
                print(f"  ✓ New best model saved! (Dice: {val_dice['mean']:.4f})")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_frequency', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_dice['mean'])
            
            # Save history
            self.save_history()
            
            print()
        
        print("\n" + "="*80)
        print(f"Training Complete!")
        print(f"Best Epoch: {self.best_epoch + 1} | Best Val Dice: {self.best_val_dice:.4f}")
        print("="*80 + "\n")
    
    def save_checkpoint(self, filename, epoch, val_dice):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_dice': val_dice,
            'history': self.history,
            'config': self.config
        }
        torch.save(checkpoint, os.path.join(self.config['checkpoint_dir'], filename))
    
    def save_history(self):
        history_path = os.path.join(self.config['log_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

def main():
    # Configuration
    config = {
        'data_dir': './processed_data',
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
        'batch_size': 1,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_workers': 4,
        'save_frequency': 10,
        'train_split': 0.8
    }
    
    # Load dataset
    all_files = [f.replace('.npz', '') for f in os.listdir(config['data_dir']) if f.endswith('.npz')]
    np.random.seed(42)
    np.random.shuffle(all_files)
    
    split_idx = int(len(all_files) * config['train_split'])
    train_ids = all_files[:split_idx]
    val_ids = all_files[split_idx:]
    
    train_dataset = CTDataset(config['data_dir'], train_ids)
    val_dataset = CTDataset(config['data_dir'], val_ids)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Initialize model
    model = MultiDecoderUNet3D(in_channels=1, num_classes_per_organ=1)
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()

if __name__ == "__main__":
    main()
