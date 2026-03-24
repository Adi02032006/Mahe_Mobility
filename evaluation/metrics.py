"""
evaluation/metrics.py
---------------------
minADE  : average displacement error of best mode
minFDE  : final displacement error of best mode
MissRate: fraction of samples where best-mode FDE > threshold (2m)
"""

import torch
import numpy as np


def min_ade(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """preds (B,K,T,2), target (B,T,2)"""
    tgt  = target.unsqueeze(1).expand_as(preds)
    errs = (preds - tgt).norm(dim=-1).mean(dim=-1)   # (B,K)
    return errs.min(dim=1).values.mean()


def min_fde(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """preds (B,K,T,2), target (B,T,2)"""
    tgt_f  = target[:, -1, :]                          # (B,2)
    pred_f = preds[:, :, -1, :]                        # (B,K,2)
    errs   = (pred_f - tgt_f.unsqueeze(1)).norm(dim=-1) # (B,K)
    return errs.min(dim=1).values.mean()


def miss_rate(preds: torch.Tensor, target: torch.Tensor,
              threshold: float = 2.0) -> torch.Tensor:
    """Fraction of samples where best-mode FDE > threshold."""
    tgt_f  = target[:, -1, :]
    pred_f = preds[:, :, -1, :]
    errs   = (pred_f - tgt_f.unsqueeze(1)).norm(dim=-1)
    best   = errs.min(dim=1).values
    return (best > threshold).float().mean()


def evaluate_batch(preds: torch.Tensor, targets: torch.Tensor,
                   miss_thresh: float = 2.0) -> dict:
    return {
        "minADE":    min_ade(preds, targets).item(),
        "minFDE":    min_fde(preds, targets).item(),
        "MissRate":  miss_rate(preds, targets, miss_thresh).item(),
    }


def aggregate_metrics(metric_list: list) -> dict:
    keys = metric_list[0].keys()
    return {k: float(np.mean([m[k] for m in metric_list])) for k in keys}
