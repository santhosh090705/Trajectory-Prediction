"""
evaluate.py
-----------
Byte Riders | MAHE Mobility Challenge 2026
Track 1: Intent & Trajectory Prediction

Evaluates a trained model checkpoint on the test split and prints:
    - minADE@K  (primary metric)
    - minFDE@K  (primary metric)
    - Miss Rate  (FDE > 2 metres)
    - Per-horizon ADE (breakdown by time step)

Usage:
    python evaluate.py --dataroot ./data/nuscenes \
                       --checkpoint ./checkpoints/best_model.pt
"""

import os
import argparse
import numpy as np
import torch
from collections import defaultdict

from model import TrajectoryPredictor, PRED_LEN, K
from data_loader import get_dataloader

MISS_THRESHOLD = 2.0   # metres – standard nuScenes threshold


# ─────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────

def compute_min_ade(pred, target):
    """
    minADE@K – mean over time of min-over-modes L2 distance.

    Args:
        pred   : (B, K, T, 2)
        target : (B, T, 2)

    Returns:
        per_sample minADE : (B,)
    """
    target_exp   = target.unsqueeze(1).expand_as(pred)     # (B, K, T, 2)
    dist         = torch.norm(pred - target_exp, dim=-1)   # (B, K, T)
    ade_per_mode = dist.mean(dim=-1)                       # (B, K)
    min_ade, _   = ade_per_mode.min(dim=1)                 # (B,)
    return min_ade


def compute_min_fde(pred, target):
    """
    minFDE@K – L2 distance at final timestep, best mode.

    Args:
        pred   : (B, K, T, 2)
        target : (B, T, 2)

    Returns:
        per_sample minFDE : (B,)
    """
    pred_final   = pred[:, :, -1, :]                          # (B, K, 2)
    target_final = target[:, -1, :]                           # (B, 2)
    dist         = torch.norm(
        pred_final - target_final.unsqueeze(1), dim=-1
    )                                                         # (B, K)
    min_fde, _   = dist.min(dim=1)                            # (B,)
    return min_fde


def compute_miss_rate(pred, target, threshold=MISS_THRESHOLD):
    """
    Fraction of samples where minFDE > threshold.

    Returns: scalar (float)
    """
    min_fde = compute_min_fde(pred, target)
    return (min_fde > threshold).float().mean().item()


def compute_per_horizon_ade(pred, target):
    """
    ADE broken down per future timestep (best mode).

    Returns: (T,) numpy array
    """
    target_exp   = target.unsqueeze(1).expand_as(pred)     # (B, K, T, 2)
    dist         = torch.norm(pred - target_exp, dim=-1)   # (B, K, T)
    ade_per_mode = dist.mean(dim=-1)                       # (B, K)
    best_mode    = ade_per_mode.argmin(dim=1)              # (B,)

    B, K_, T, _ = pred.shape
    best_pred    = pred[torch.arange(B), best_mode]        # (B, T, 2)
    horizon_dist = torch.norm(
        best_pred - target, dim=-1
    ).mean(dim=0)                                          # (T,)

    return horizon_dist.cpu().numpy()


# ─────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(model, loader, device):
    model.eval()

    all_min_ade   = []
    all_min_fde   = []
    all_miss      = []
    horizon_acc   = np.zeros(PRED_LEN)
    n_batches     = len(loader)

    for i, (obs, fut, neigh, _) in enumerate(loader):
        obs   = obs.to(device)
        fut   = fut.to(device)
        neigh = neigh.to(device)

        pred_trajs, _ = model(obs, neigh)     # (B, K, T, 2)

        all_min_ade.append(compute_min_ade(pred_trajs, fut).cpu())
        all_min_fde.append(compute_min_fde(pred_trajs, fut).cpu())
        all_miss.append(
            (compute_min_fde(pred_trajs, fut) > MISS_THRESHOLD)
            .float().cpu()
        )
        horizon_acc += compute_per_horizon_ade(pred_trajs, fut)

        if (i + 1) % 20 == 0:
            print(f"  Evaluated {i+1}/{n_batches} batches...", flush=True)

    all_min_ade = torch.cat(all_min_ade)
    all_min_fde = torch.cat(all_min_fde)
    all_miss    = torch.cat(all_miss)
    horizon_acc = horizon_acc / n_batches

    metrics = {
        'minADE@K'  : all_min_ade.mean().item(),
        'minFDE@K'  : all_min_fde.mean().item(),
        'MissRate'  : all_miss.mean().item(),
        'per_horizon_ade': horizon_acc,
        'n_samples' : len(all_min_ade),
    }
    return metrics


# ─────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────

def print_report(metrics, K=K, pred_len=PRED_LEN, freq=2):
    """Pretty-print evaluation results."""
    print("\n" + "="*55)
    print("  BYTE RIDERS – Evaluation Report")
    print("  Track 1: Intent & Trajectory Prediction")
    print("="*55)
    print(f"  Samples evaluated : {metrics['n_samples']}")
    print(f"  K (modes)         : {K}")
    print(f"  Prediction horizon: {pred_len/freq:.1f}s ({pred_len} steps)")
    print("-"*55)
    print(f"  minADE@{K}          : {metrics['minADE@K']:.4f} m")
    print(f"  minFDE@{K}          : {metrics['minFDE@K']:.4f} m")
    print(f"  Miss Rate (>{MISS_THRESHOLD}m) : "
          f"{metrics['MissRate']*100:.2f} %")
    print("-"*55)
    print("  Per-Horizon ADE breakdown:")
    for t, ade_t in enumerate(metrics['per_horizon_ade']):
        sec = (t + 1) / freq
        print(f"    t={sec:.1f}s → ADE = {ade_t:.4f} m")
    print("="*55)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate Byte Riders Trajectory Predictor'
    )
    p.add_argument('--dataroot',    type=str,
                   default='./data/nuscenes')
    p.add_argument('--checkpoint',  type=str,
                   default='./checkpoints/best_model.pt')
    p.add_argument('--split',       type=str, default='test',
                   choices=['train', 'val', 'test'])
    p.add_argument('--batch_size',  type=int, default=64)
    p.add_argument('--num_workers', type=int, default=2)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Evaluate] Device    : {device}")
    print(f"[Evaluate] Split     : {args.split}")
    print(f"[Evaluate] Checkpoint: {args.checkpoint}")

    # ── Load checkpoint ──────────────────────
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt.get('args', {})

    # ── Build model ──────────────────────────
    model = TrajectoryPredictor(
        hidden_dim=cfg.get('hidden_dim', 64),
        num_layers=cfg.get('num_layers', 1),
        dropout=0.0,                   # no dropout during evaluation
        K=cfg.get('K', K),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    print(f"[Evaluate] Loaded model from epoch {ckpt.get('epoch','?')}")

    # ── Data loader ──────────────────────────
    loader = get_dataloader(
        args.dataroot, split=args.split,
        batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    # ── Run evaluation ───────────────────────
    metrics = run_evaluation(model, loader, device)
    print_report(metrics)

    return metrics


if __name__ == '__main__':
    main()
