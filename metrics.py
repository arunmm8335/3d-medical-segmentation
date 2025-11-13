import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

def compute_dice_score(pred, target, smooth=1e-6):
    """
    Compute Dice coefficient
    Args:
        pred: predicted mask [B, C, D, H, W]
        target: ground truth mask [B, C, D, H, W]
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def compute_iou(pred, target, smooth=1e-6):
    """
    Compute Intersection over Union
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def compute_hausdorff_distance(pred, target, spacing=(1.0, 1.0, 1.0)):
    """
    Compute Hausdorff distance between two binary masks
    Args:
        pred: predicted mask [B, 1, D, H, W] numpy array
        target: ground truth mask [B, 1, D, H, W] numpy array
        spacing: voxel spacing in mm
    Returns:
        hausdorff distance in mm
    """
    if pred.shape[0] > 1:
        # Compute for batch
        hd_list = []
        for i in range(pred.shape[0]):
            hd = compute_hausdorff_distance(pred[i:i+1], target[i:i+1], spacing)
            hd_list.append(hd)
        return np.mean(hd_list)
    
    pred = pred.squeeze()
    target = target.squeeze()
    
    # If either mask is empty
    if pred.sum() == 0 or target.sum() == 0:
        if pred.sum() == target.sum():
            return 0.0
        else:
            return 999.0  # Large value for empty masks
    
    # Compute distance transforms
    pred_dt = distance_transform_edt(1 - pred, sampling=spacing)
    target_dt = distance_transform_edt(1 - target, sampling=spacing)
    
    # Get surface points
    pred_surface = (pred > 0)
    target_surface = (target > 0)
    
    # Compute distances from pred surface to target
    distances_pred_to_target = pred_dt[pred_surface]
    # Compute distances from target surface to pred
    distances_target_to_pred = target_dt[target_surface]
    
    # Hausdorff distance
    hd = max(distances_pred_to_target.max(), distances_target_to_pred.max())
    
    return float(hd)

def compute_hausdorff_95(pred, target, spacing=(1.0, 1.0, 1.0)):
    """
    Compute 95th percentile Hausdorff distance
    More robust to outliers than standard HD
    """
    if pred.shape[0] > 1:
        hd95_list = []
        for i in range(pred.shape[0]):
            hd95 = compute_hausdorff_95(pred[i:i+1], target[i:i+1], spacing)
            hd95_list.append(hd95)
        return np.mean(hd95_list)
    
    pred = pred.squeeze()
    target = target.squeeze()
    
    if pred.sum() == 0 or target.sum() == 0:
        if pred.sum() == target.sum():
            return 0.0
        else:
            return 999.0
    
    pred_dt = distance_transform_edt(1 - pred, sampling=spacing)
    target_dt = distance_transform_edt(1 - target, sampling=spacing)
    
    pred_surface = (pred > 0)
    target_surface = (target > 0)
    
    distances_pred_to_target = pred_dt[pred_surface]
    distances_target_to_pred = target_dt[target_surface]
    
    # 95th percentile
    hd95 = max(
        np.percentile(distances_pred_to_target, 95),
        np.percentile(distances_target_to_pred, 95)
    )
    
    return float(hd95)

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, D, H, W]
            target: [B, C, D, H, W]
        """
        pred = pred.contiguous()
        target = target.contiguous()
        
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """Combined Dice and Focal Loss"""
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal

def compute_sensitivity(pred, target, smooth=1e-6):
    """
    Sensitivity (Recall, True Positive Rate)
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    TP = (pred * target).sum()
    FN = ((1 - pred) * target).sum()
    
    sensitivity = (TP + smooth) / (TP + FN + smooth)
    return sensitivity.item()

def compute_specificity(pred, target, smooth=1e-6):
    """
    Specificity (True Negative Rate)
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    TN = ((1 - pred) * (1 - target)).sum()
    FP = (pred * (1 - target)).sum()
    
    specificity = (TN + smooth) / (TN + FP + smooth)
    return specificity.item()

def compute_precision(pred, target, smooth=1e-6):
    """
    Precision (Positive Predictive Value)
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    
    precision = (TP + smooth) / (TP + FP + smooth)
    return precision.item()

def compute_f1_score(pred, target):
    """
    F1 Score (harmonic mean of precision and recall)
    """
    precision = compute_precision(pred, target)
    sensitivity = compute_sensitivity(pred, target)
    
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-6)
    return f1

def compute_all_metrics(pred, target, spacing=(1.0, 1.0, 1.0)):
    """
    Compute all metrics for evaluation
    Args:
        pred: [B, C, D, H, W] tensor
        target: [B, C, D, H, W] tensor
    Returns:
        dict with all metrics
    """
    metrics = {}
    
    # Convert to numpy for HD computation
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # Compute metrics for each channel (organ)
    organs = ['prostate', 'bladder', 'rectum']
    for idx, organ in enumerate(organs):
        pred_organ = pred[:, idx:idx+1]
        target_organ = target[:, idx:idx+1]
        
        pred_organ_np = pred_np[:, idx:idx+1]
        target_organ_np = target_np[:, idx:idx+1]
        
        metrics[f'{organ}_dice'] = compute_dice_score(pred_organ, target_organ)
        metrics[f'{organ}_iou'] = compute_iou(pred_organ, target_organ)
        metrics[f'{organ}_sensitivity'] = compute_sensitivity(pred_organ, target_organ)
        metrics[f'{organ}_specificity'] = compute_specificity(pred_organ, target_organ)
        metrics[f'{organ}_precision'] = compute_precision(pred_organ, target_organ)
        metrics[f'{organ}_f1'] = compute_f1_score(pred_organ, target_organ)
        metrics[f'{organ}_hd'] = compute_hausdorff_distance(pred_organ_np, target_organ_np, spacing)
        metrics[f'{organ}_hd95'] = compute_hausdorff_95(pred_organ_np, target_organ_np, spacing)
    
    # Compute mean metrics
    metrics['mean_dice'] = np.mean([metrics[f'{o}_dice'] for o in organs])
    metrics['mean_iou'] = np.mean([metrics[f'{o}_iou'] for o in organs])
    metrics['mean_hd'] = np.mean([metrics[f'{o}_hd'] for o in organs])
    metrics['mean_hd95'] = np.mean([metrics[f'{o}_hd95'] for o in organs])
    
    return metrics
