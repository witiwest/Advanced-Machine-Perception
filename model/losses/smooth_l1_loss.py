from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from .losses_utils import weighted_loss


@weighted_loss
def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.0) -> Tensor:
    """Smooth L1 loss (Huber loss) as in Fast R-CNN.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.
        beta (float): Transition point from L1 to L2 loss.

    Returns:
        Tensor: Element-wise loss.
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    return F.smooth_l1_loss(pred, target, beta=beta, reduction='none')


class SmoothL1Loss(nn.Module):
    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 beta: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.beta = beta

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * smooth_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor, beta=self.beta)
        return loss
