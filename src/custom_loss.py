import torch
import torch.nn as nn
from torch.nn import functional as F

class MaskedMSELoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Computes masked MSE loss. If mask is None, falls back to standard MSE.
        Args:
            input (Tensor): Predicted tensor.
            target (Tensor): Ground truth tensor (same shape as input).
            mask:   (B, T) binary {0,1} where 1 means "include in loss"

        Returns:
            Tensor: Scalar loss if reduction != "none", else elementwise loss.
        """
        if mask is None:
            # fallback: behave like standard MSELoss
            return F.mse_loss(input, target, reduction=self.reduction)

        # Expand mask to broadcast over C,H,W
        mask = mask[:, None, :, None, None]  # (B,1,T,1,1)
        B, C, T, H, W = input.shape

        diff = (input - target) ** 2
        masked_diff = diff * mask  # masked out entries = 0

        if self.reduction == "mean":
            return masked_diff.sum() / (mask.sum() * C * H * W).clamp(min=1)
        elif self.reduction == "sum":
            return masked_diff.sum()
        elif self.reduction == "none":
            return masked_diff
