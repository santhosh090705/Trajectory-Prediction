"""
train.py
--------
Byte Riders | MAHE Mobility Challenge 2026
Track 1: Intent & Trajectory Prediction

Training script for the TrajectoryPredictor model.

Usage:
    python train.py --dataroot ./data/nuscenes --epochs 50 --batch_size 64

Loss:
    L = λ_reg * MSE(best_pred, fut) + λ_cls * NLL(log_probs, best_mode)

    - MSE on the best-matching mode (minADE supervision)
    - NLL to reinforce the probability of the correct mode
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import TrajectoryPredictor, OBS_LEN, PRED_LEN, K
from data_loader import get_dataloader


# ─────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────

def ade(pred, target):
    """
    Average Displacement Error.
    pred   : (B, K, T, 2)
    target : (B, T, 2)
    Returns: scalar ADE averaged over batch
    """
    target_exp = target.unsqueeze(1).expand_as(pred)         # (B, K, T, 2)
    dist       = torch.norm(pred - target_exp, dim=-1)       # (B, K, T)
    ade_per_mode = dist.mean(dim=-1)                         # (B, K)
    min_ade, _   = ade_per_mode.min(dim=1)                   # (B,)
    return min_ade.mean()


def fde(pred, target):
    """
    Final Displacement Error.
    pred   : (B, K, T, 2)
    target : (B, T, 2)
    Returns: scalar FDE averaged over batch
    """
    target_final = target[:, -1, :]                          # (B, 2)
    pred_final   = pred[:, :, -1, :]                         # (B, K, 2)
    dist = torch.norm(
        pred_final - target_final.unsqueeze(1), dim=-1
    )                                                        # (B, K)
    min_fde, _ = dist.min(dim=1)                             # (B,)
    return min_fde.mean()


def trajectory_loss(pred_trajs, log_probs, fut,
                    lambda_reg=1.0, lambda_cls=0.5):
    """
    Combined regression + classification loss.

    Args:
        pred_trajs  : (B, K, T, 2)
        log_probs   : (B, K)
        fut         : (B, T, 2)
        lambda_reg  : weight for regression MSE
        lambda_cls  : weight for NLL mode classification

    Returns:
        total_loss, ade_val, fde_val
    """
    B, K_, T, _ = pred_trajs.shape
    fut_exp = fut.unsqueeze(1).expand_as(pred_trajs)         # (B, K, T, 2)

    # Find best mode (lowest ADE) per sample
    dist_per_mode = torch.norm(
        pred_trajs - fut_exp, dim=-1
    ).mean(dim=-1)                                           # (B, K)
    best_mode = dist_per_mode.argmin(dim=1)                  # (B,)

    # Regression loss: MSE between best mode prediction and ground truth
    best_pred = pred_trajs[torch.arange(B), best_mode]      # (B, T, 2)
    reg_loss  = nn.functional.mse_loss(best_pred, fut)

    # Classification loss: NLL for the best mode
    cls_loss  = nn.functional.nll_loss(log_probs, best_mode)

    total_loss = lambda_reg * reg_loss + lambda_cls * cls_loss

    with torch.no_grad():
        ade_val = ade(pred_trajs, fut).item()
        fde_val = fde(pred_trajs, fut).item()

    return total_loss, ade_val, fde_val


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_ade  = 0.0
    total_fde  = 0.0
    n_batches  = len(loader)

    for batch_idx, (obs, fut, neigh, _) in enumerate(loader):
        obs   = obs.to(device)       # (B, OBS_LEN, 2)
        fut   = fut.to(device)       # (B, PRED_LEN, 2)
        neigh = neigh.to(device)     # (B, N, OBS_LEN, 2)

        optimizer.zero_grad()
        pred_trajs, log_probs = model(obs, neigh)
        loss, ade_val, fde_val = trajectory_loss(
            pred_trajs, log_probs, fut
        )
        loss.backward()

        # Gradient clipping for stable training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ade  += ade_val
        total_fde  += fde_val

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches:
            print(f"  Epoch {epoch} [{batch_idx+1}/{n_batches}] "
                  f"Loss: {loss.item():.4f} | "
                  f"ADE: {ade_val:.4f} | FDE: {fde_val:.4f}")

    return (total_loss / n_batches,
            total_ade  / n_batches,
            total_fde  / n_batches)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_ade = 0.0
    total_fde = 0.0
    n_batches = len(loader)

    for obs, fut, neigh, _ in loader:
        obs   = obs.to(device)
        fut   = fut.to(device)
        neigh = neigh.to(device)

        pred_trajs, log_probs = model(obs, neigh)
        total_ade += ade(pred_trajs, fut).item()
        total_fde += fde(pred_trajs, fut).item()

    return total_ade / n_batches, total_fde / n_batches


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Byte Riders Trajectory Predictor"
    )
    p.add_argument('--dataroot',    type=str,   default='./data/nuscenes',
                   help='Path to nuScenes dataset root')
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--batch_size',  type=int,   default=64)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--hidden_dim',  type=int,   default=64)
    p.add_argument('--num_layers',  type=int,   default=1)
    p.add_argument('--dropout',     type=float, default=0.1)
    p.add_argument('--K',           type=int,   default=3)
    p.add_argument('--num_workers', type=int,   default=2)
    p.add_argument('--save_dir',    type=str,   default='./checkpoints')
    p.add_argument('--resume',      type=str,   default=None,
                   help='Path to checkpoint to resume from')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Train] Device: {device}")
    print(f"[Train] Config: {vars(args)}\n")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Data ─────────────────────────────────
    train_loader = get_dataloader(
        args.dataroot, split='train',
        batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )
    val_loader = get_dataloader(
        args.dataroot, split='val',
        batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    # ── Model ────────────────────────────────
    model = TrajectoryPredictor(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        K=args.K,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Model parameters: {total_params:,}\n")

    # ── Optimiser & Scheduler ─────────────────
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=5
    )

    # ── Optional Resume ───────────────────────
    start_epoch = 1
    best_ade    = float('inf')

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_ade    = ckpt.get('best_ade', float('inf'))
        print(f"[Train] Resumed from epoch {ckpt['epoch']} "
              f"(best ADE: {best_ade:.4f})")

    # ── Training Loop ─────────────────────────
    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\n{'='*55}")
        print(f"  Epoch {epoch}/{args.epochs}")
        print(f"{'='*55}")

        train_loss, train_ade, train_fde = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        val_ade, val_fde = evaluate(model, val_loader, device)
        scheduler.step(val_ade)

        elapsed = time.time() - t0
        print(f"\n  ── Epoch {epoch} Summary ──────────────────")
        print(f"  Train Loss : {train_loss:.4f}")
        print(f"  Train ADE  : {train_ade:.4f}  |  Train FDE : {train_fde:.4f}")
        print(f"  Val   ADE  : {val_ade:.4f}  |  Val   FDE : {val_fde:.4f}")
        print(f"  Time       : {elapsed:.1f}s")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_ade': train_ade,
            'train_fde': train_fde,
            'val_ade': val_ade,
            'val_fde': val_fde,
        })

        # Save checkpoint every epoch
        ckpt_path = os.path.join(args.save_dir, f'epoch_{epoch:03d}.pt')
        torch.save({
            'epoch'    : epoch,
            'model'    : model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_ade'  : val_ade,
            'val_fde'  : val_fde,
            'best_ade' : best_ade,
            'args'     : vars(args),
        }, ckpt_path)

        # Save best model separately
        if val_ade < best_ade:
            best_ade = val_ade
            best_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch'  : epoch,
                'model'  : model.state_dict(),
                'val_ade': val_ade,
                'val_fde': val_fde,
                'args'   : vars(args),
            }, best_path)
            print(f"  ★ New best ADE: {best_ade:.4f} → saved to {best_path}")

    print(f"\n[Train] Finished. Best Val ADE: {best_ade:.4f}")
    return history


if __name__ == '__main__':
    main()
