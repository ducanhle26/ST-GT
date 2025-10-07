"""
Loss Functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImbalancedFocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha: float = 0.3, gamma: float = 2.0, pos_weight: float = None):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter
            pos_weight: Additional weight for positive class
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            logits: Model predictions (before sigmoid) [B]
            targets: Ground truth labels [B]
            
        Returns:
            loss: Scalar loss value
        """
        if self.pos_weight is not None:
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight, reduction='none'
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Compute focal weight
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()