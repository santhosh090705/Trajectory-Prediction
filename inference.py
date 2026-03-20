"""
inference.py
------------
Byte Riders | MAHE Mobility Challenge 2026
Track 1: Intent & Trajectory Prediction

Runs inference on a single scene or a batch and saves:
    - Predicted trajectories as a JSON file
    - Visualisation PNG (top-3 predicted paths over observed path)

Usage:
    python inference.py --dataroot ./data/nuscenes \
                        --checkpoint ./checkpoints/best_model.pt \
                        --num_scenes 5 \
                        --output_dir ./outputs
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')                # headless rendering
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from model import TrajectoryPredictor, OBS_LEN, PRED_LEN, K
from data_loader import get_dataloader


# ─────────────────────────────────────────────
# Colour palette for K modes
# ─────────────────────────────────────────────
MODE_COLORS  = ['#E74C3C', '#3498DB', '#2ECC71']   # red, blue, green
MODE_LABELS  = ['Mode 1 (best)', 'Mode 2', 'Mode 3']
OBS_COLOR    = '#F39C12'    # orange
GT_COLOR     = '#9B59B6'    # purple


# ─────────────────────────────────────────────
# Single-scene visualisation
# ─────────────────────────────────────────────

def visualise_scene(obs, fut, pred_trajs, log_probs, scene_idx,
                    output_dir, freq=2):
    """
    Plot observed path, ground truth future, and K predicted trajectories.

    Args:
        obs        : (OBS_LEN, 2)  – normalised observed positions
        fut        : (PRED_LEN, 2) – normalised ground truth future
        pred_trajs : (K, PRED_LEN, 2)
        log_probs  : (K,)          – log-probabilities of each mode
        scene_idx  : int
        output_dir : str
        freq       : Hz (2 for nuScenes)
    """
    probs = np.exp(log_probs)                    # (K,)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    # ── Observed trajectory ──────────────────
    ax.plot(obs[:, 0], obs[:, 1],
            color=OBS_COLOR, linewidth=2.5,
            marker='o', markersize=5,
            label='Observed (2s)', zorder=5)
    ax.scatter(obs[-1, 0], obs[-1, 1],
               color=OBS_COLOR, s=80, zorder=6)

    # ── Ground truth future ──────────────────
    gt_full = np.vstack([obs[-1:], fut])
    ax.plot(gt_full[:, 0], gt_full[:, 1],
            color=GT_COLOR, linewidth=2.5,
            linestyle='--', marker='s', markersize=4,
            label='Ground Truth (3s)', zorder=5)

    # ── Predicted modes ──────────────────────
    for k in range(K):
        traj_k = pred_trajs[k]                   # (T, 2)
        full_k = np.vstack([obs[-1:], traj_k])
        prob_k = probs[k]
        color  = MODE_COLORS[k % len(MODE_COLORS)]
        label  = f'{MODE_LABELS[k % len(MODE_LABELS)]} (p={prob_k:.2f})'

        ax.plot(full_k[:, 0], full_k[:, 1],
                color=color, linewidth=2.0,
                linestyle='-', alpha=0.85,
                marker='^', markersize=4,
                label=label, zorder=4)

    # ── Styling ───────────────────────────────
    ax.set_title(f'Scene {scene_idx} – Top-{K} Trajectory Predictions',
                 color='white', fontsize=13, pad=12)
    ax.set_xlabel('x (m)', color='white')
    ax.set_ylabel('y (m)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    legend = ax.legend(loc='upper left', fontsize=9,
                       facecolor='#2c2c54', labelcolor='white',
                       edgecolor='#444')

    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, color='#333', linewidth=0.5)

    # Time labels on GT
    for t, (x, y) in enumerate(fut):
        sec = (t + 1) / freq
        ax.annotate(f'{sec:.1f}s', (x, y),
                    textcoords='offset points',
                    xytext=(5, 5), fontsize=7,
                    color=GT_COLOR, alpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'scene_{scene_idx:04d}.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────
# Inference runner
# ─────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device, num_scenes, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualisations')
    os.makedirs(vis_dir, exist_ok=True)

    results   = []
    scene_idx = 0

    print(f"\n[Inference] Running on {num_scenes} scene(s)...")

    for obs_batch, fut_batch, neigh_batch, origin_batch in loader:
        if scene_idx >= num_scenes:
            break

        obs_batch   = obs_batch.to(device)
        fut_batch   = fut_batch.to(device)
        neigh_batch = neigh_batch.to(device)

        pred_trajs, log_probs = model(obs_batch, neigh_batch)
        # pred_trajs : (B, K, T, 2) — normalised coords
        # Denormalise using origin
        origin_np = origin_batch.numpy()          # (B, 2)

        B = obs_batch.size(0)
        for b in range(B):
            if scene_idx >= num_scenes:
                break

            obs_np  = obs_batch[b].cpu().numpy()          # (OBS_LEN, 2)
            fut_np  = fut_batch[b].cpu().numpy()          # (PRED_LEN, 2)
            preds_np = pred_trajs[b].cpu().numpy()        # (K, PRED_LEN, 2)
            lp_np   = log_probs[b].cpu().numpy()          # (K,)
            orig    = origin_np[b]                        # (2,)

            # Denormalise (add back origin offset)
            obs_world   = obs_np  + orig
            fut_world   = fut_np  + orig
            preds_world = preds_np + orig[np.newaxis, np.newaxis, :]

            # Visualise (normalised coords are cleaner to plot)
            vis_path = visualise_scene(
                obs_np, fut_np, preds_np, lp_np,
                scene_idx, vis_dir
            )

            # Sort modes by probability (highest first)
            probs     = np.exp(lp_np)
            order     = np.argsort(-probs)

            results.append({
                'scene_idx'  : scene_idx,
                'origin'     : orig.tolist(),
                'observed'   : obs_world.tolist(),
                'ground_truth': fut_world.tolist(),
                'predictions': [
                    {
                        'mode'        : int(order[k]),
                        'probability' : float(probs[order[k]]),
                        'trajectory'  : preds_world[order[k]].tolist(),
                    }
                    for k in range(K)
                ],
                'visualisation': vis_path,
            })

            print(f"  Scene {scene_idx:04d} → "
                  f"probs={[f'{p:.3f}' for p in probs[order]]} | "
                  f"saved: {os.path.basename(vis_path)}")
            scene_idx += 1

    # Save all results to JSON
    json_path = os.path.join(output_dir, 'predictions.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Inference] Done. {scene_idx} scenes processed.")
    print(f"[Inference] Predictions saved → {json_path}")
    print(f"[Inference] Visualisations  → {vis_dir}/")
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Byte Riders – Trajectory Inference'
    )
    p.add_argument('--dataroot',    type=str,
                   default='./data/nuscenes')
    p.add_argument('--checkpoint',  type=str,
                   default='./checkpoints/best_model.pt')
    p.add_argument('--split',       type=str, default='test',
                   choices=['train', 'val', 'test'])
    p.add_argument('--num_scenes',  type=int, default=10,
                   help='Number of scenes to run inference on')
    p.add_argument('--batch_size',  type=int, default=16)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--output_dir',  type=str, default='./outputs')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Inference] Device     : {device}")
    print(f"[Inference] Checkpoint : {args.checkpoint}")
    print(f"[Inference] Split      : {args.split}")
    print(f"[Inference] Scenes     : {args.num_scenes}")

    # ── Load checkpoint ──────────────────────
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt.get('args', {})

    # ── Model ────────────────────────────────
    model = TrajectoryPredictor(
        hidden_dim=cfg.get('hidden_dim', 64),
        num_layers=cfg.get('num_layers', 1),
        dropout=0.0,
        K=cfg.get('K', K),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    print(f"[Inference] Model loaded from epoch {ckpt.get('epoch','?')}")

    # ── Data ─────────────────────────────────
    loader = get_dataloader(
        args.dataroot, split=args.split,
        batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    # ── Run ──────────────────────────────────
    results = run_inference(
        model, loader, device,
        num_scenes=args.num_scenes,
        output_dir=args.output_dir,
    )
    return results


if __name__ == '__main__':
    main()
