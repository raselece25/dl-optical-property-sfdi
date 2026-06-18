"""
train.py
========
Training script for deep learning SFDI optical property extraction.

Supports:
  - U-Net, Attention U-Net, lightweight CNN (from unet_model.py)
  - Custom SFDI dataset (from dataset.py)
  - Mixed loss: MAE + relative MAE (scale-invariant)
  - Cosine-annealing LR schedule
  - TensorBoard logging
  - Checkpointing (best val loss)

Usage:
    python train.py --arch unet --epochs 100 --batch 16 --lr 1e-3
    python train.py --arch attention_unet --data /path/to/dataset --cuda

Author : Rasel Ahmmed | rasel.ahmmed@stonybrook.edu
"""

import argparse
import time
import logging
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from unet_model import build_model
from dataset import SFDIDataset, SFDISimulatedDataset

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ── Loss Functions ─────────────────────────────────────────────────────────────

class SFDILoss(nn.Module):
    """
    Combined loss for optical property regression:
        L = λ_mae * L_MAE + λ_rel * L_RelMAE

    L_MAE    = mean |pred - target|
    L_RelMAE = mean |pred - target| / (|target| + ε)  – scale-invariant

    Both terms are computed on log-transformed values for numerical stability
    since μa and μs' span multiple orders of magnitude.
    """

    def __init__(self, lambda_mae: float = 0.5, lambda_rel: float = 0.5,
                 eps: float = 1e-6, use_log: bool = True):
        super().__init__()
        self.lambda_mae = lambda_mae
        self.lambda_rel = lambda_rel
        self.eps        = eps
        self.use_log    = use_log

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_log:
            pred   = torch.log(pred   + self.eps)
            target = torch.log(target + self.eps)

        diff    = torch.abs(pred - target)
        mae     = diff.mean()
        rel_mae = (diff / (torch.abs(target) + self.eps)).mean()

        return self.lambda_mae * mae + self.lambda_rel * rel_mae


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(pred: torch.Tensor, target: torch.Tensor,
                    eps: float = 1e-8) -> Dict[str, float]:
    """
    Per-channel metrics for μa (ch 0) and μs' (ch 1).
    Returns RMSE, MAPE, and R² for each channel.
    """
    pred   = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    metrics = {}

    for i, name in enumerate(['mu_a', 'mu_s_prime']):
        p = pred[:, i].ravel()
        t = target[:, i].ravel()

        rmse = np.sqrt(np.mean((p - t)**2))
        mape = np.mean(np.abs(p - t) / (np.abs(t) + eps)) * 100
        ss_res = np.sum((t - p)**2)
        ss_tot = np.sum((t - t.mean())**2)
        r2     = 1 - ss_res / (ss_tot + eps)

        metrics[f'{name}_rmse'] = rmse
        metrics[f'{name}_mape'] = mape
        metrics[f'{name}_r2']   = r2

    return metrics


# ── Training Loop ──────────────────────────────────────────────────────────────

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: str) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        preds = model(inputs)
        loss  = criterion(preds, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: str) -> Tuple[float, Dict]:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for inputs, targets in loader:
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds   = model(inputs)
        total_loss += criterion(preds, targets).item()
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    val_loss = total_loss / len(loader)
    metrics  = compute_metrics(
        torch.cat(all_preds), torch.cat(all_targets))

    return val_loss, metrics


def train(args: argparse.Namespace):
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    if args.data == 'simulated':
        logger.info(f"Using synthetic SFDI dataset ({args.n_samples} samples)")
        dataset = SFDISimulatedDataset(
            n_samples=args.n_samples,
            image_size=args.imsize,
            n_freqs=args.n_freqs)
    else:
        logger.info(f"Loading real dataset from: {args.data}")
        dataset = SFDIDataset(args.data, image_size=args.imsize)

    n_val   = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=args.workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True)

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ── Model ────────────────────────────────────────────────────────────────
    in_ch = 2 * args.n_freqs   # AC + DC per frequency
    model = build_model(args.arch, in_channels=in_ch, out_channels=2,
                        base_features=args.features).to(device)

    # ── Optimizer & Schedule ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = SFDILoss(lambda_mae=0.5, lambda_rel=0.5)

    # ── TensorBoard ──────────────────────────────────────────────────────────
    writer = None
    if HAS_TB and args.logdir:
        writer = SummaryWriter(args.logdir)

    # ── Checkpoint dir ───────────────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    # ── Training Loop ────────────────────────────────────────────────────────
    logger.info(f"Starting training: {args.epochs} epochs, arch={args.arch}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss           = train_one_epoch(model, train_loader,
                                               optimizer, criterion, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Epoch [{epoch:03d}/{args.epochs}]  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"μa_R²={val_metrics['mu_a_r2']:.3f}  "
            f"μs'_R²={val_metrics['mu_s_prime_r2']:.3f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"({elapsed:.1f}s)")

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val',   val_loss,   epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f'Metrics/{k}', v, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = ckpt_dir / f'best_{args.arch}.pth'
            torch.save({
                'epoch':      epoch,
                'arch':       args.arch,
                'state_dict': model.state_dict(),
                'val_loss':   val_loss,
                'metrics':    val_metrics,
                'args':       vars(args),
            }, ckpt_path)
            logger.info(f"  ✓ Saved best checkpoint → {ckpt_path}")

    if writer:
        writer.close()

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return model


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train U-Net/CNN for SFDI optical property extraction')

    # Data
    parser.add_argument('--data', type=str, default='simulated',
                        help='Path to dataset directory or "simulated"')
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Number of synthetic training samples')
    parser.add_argument('--imsize', type=int, default=128,
                        help='Input image size (H = W)')
    parser.add_argument('--n_freqs', type=int, default=2,
                        help='Number of spatial frequencies')

    # Model
    parser.add_argument('--arch', type=str, default='unet',
                        choices=['unet', 'attention_unet', 'cnn'])
    parser.add_argument('--features', type=int, default=32,
                        help='Base feature maps in U-Net encoder')

    # Training
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--batch',       type=int,   default=8)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--weight_decay',type=float, default=1e-4)
    parser.add_argument('--workers',     type=int,   default=4)

    # Infrastructure
    parser.add_argument('--cuda',     action='store_true')
    parser.add_argument('--logdir',   type=str, default='runs/')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
