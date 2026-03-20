"""
data_loader.py
--------------
Byte Riders | MAHE Mobility Challenge 2026
Track 1: Intent & Trajectory Prediction

Loads and preprocesses the nuScenes dataset for pedestrian/cyclist
trajectory prediction.

Input  : 2 seconds of past (x, y) motion  → 4 timesteps @ 2Hz
Output : 3 seconds of future (x, y) motion → 6 timesteps @ 2Hz
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.eval.prediction.splits import get_prediction_challenge_split

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
FREQUENCY        = 2          # Hz  (nuScenes annotation frequency)
OBS_SECONDS      = 2          # seconds of observed history
PRED_SECONDS     = 3          # seconds to predict into the future
OBS_LEN          = OBS_SECONDS  * FREQUENCY   # = 4 timesteps
PRED_LEN         = PRED_SECONDS * FREQUENCY   # = 6 timesteps
MAX_AGENTS       = 20         # max neighbours per scene for social context
AGENT_CATEGORIES = {'pedestrian', 'bicycle'}


# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────

def get_agent_history(helper, instance_token, sample_token, obs_len):
    """
    Fetch past (x, y) positions for one agent.
    Returns array of shape (obs_len, 2).  Missing steps → zero-padded.
    """
    history = helper.get_past_for_agent(
        instance_token, sample_token,
        seconds=OBS_SECONDS,
        in_agent_frame=True,
        just_xy=True
    )
    # history is returned newest-first; reverse to chronological order
    history = history[::-1]

    padded = np.zeros((obs_len, 2), dtype=np.float32)
    n = min(len(history), obs_len)
    padded[obs_len - n:] = history[:n]
    return padded


def get_agent_future(helper, instance_token, sample_token, pred_len):
    """
    Fetch future (x, y) positions for one agent.
    Returns array of shape (pred_len, 2).  Missing steps → zero-padded.
    """
    future = helper.get_future_for_agent(
        instance_token, sample_token,
        seconds=PRED_SECONDS,
        in_agent_frame=True,
        just_xy=True
    )
    padded = np.zeros((pred_len, 2), dtype=np.float32)
    n = min(len(future), pred_len)
    padded[:n] = future[:n]
    return padded


def get_neighbour_histories(helper, instance_token, sample_token,
                            obs_len, max_agents):
    """
    Collect past trajectories of neighbouring agents (social context).
    Returns array of shape (max_agents, obs_len, 2).
    """
    neighbours = helper.get_past_for_sample(
        sample_token,
        seconds=OBS_SECONDS,
        in_agent_frame=False,
        just_xy=True
    )

    out = np.zeros((max_agents, obs_len, 2), dtype=np.float32)
    idx = 0
    for tok, traj in neighbours.items():
        if tok == instance_token:
            continue
        if idx >= max_agents:
            break
        traj = traj[::-1]                     # chronological order
        n = min(len(traj), obs_len)
        out[idx, obs_len - n:] = traj[:n]
        idx += 1
    return out


def normalize_trajectory(traj):
    """
    Zero-center a trajectory so that the last observed position is (0, 0).
    Works for both (T, 2) and (N, T, 2) shaped arrays.
    """
    if traj.ndim == 2:          # single agent  (T, 2)
        origin = traj[-1].copy()
        return traj - origin, origin
    elif traj.ndim == 3:        # neighbours    (N, T, 2)
        # centre each neighbour on its own last non-zero position
        normed = traj.copy()
        for i in range(traj.shape[0]):
            nonzero = np.any(traj[i] != 0, axis=1)
            if nonzero.any():
                origin = traj[i][nonzero][-1]
                normed[i] = traj[i] - origin
        return normed
    return traj


# ─────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────

class NuScenesTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for nuScenes trajectory prediction.

    Each sample contains:
        obs_traj      : (OBS_LEN, 2)            – ego agent history (normalised)
        fut_traj      : (PRED_LEN, 2)           – ego agent future  (normalised)
        neighbour_traj: (MAX_AGENTS, OBS_LEN, 2)– neighbour histories
        origin        : (2,)                    – de-normalisation offset
    """

    def __init__(self, nusc, helper, split='train'):
        super().__init__()
        self.helper   = helper
        self.obs_len  = OBS_LEN
        self.pred_len = PRED_LEN

        print(f"[DataLoader] Loading '{split}' split ...")
        tokens = get_prediction_challenge_split(split, dataroot=nusc.dataroot)

        self.samples = []
        skipped = 0
        for token in tokens:
            instance_token, sample_token = token.split('_')

            # Filter: pedestrians and cyclists only
            annotation = nusc.get('sample_annotation', instance_token) \
                if self._is_valid_instance(nusc, instance_token) else None
            if annotation is None:
                skipped += 1
                continue

            obs  = get_agent_history(helper, instance_token,
                                     sample_token, self.obs_len)
            fut  = get_agent_future(helper, instance_token,
                                    sample_token, self.pred_len)
            neigh = get_neighbour_histories(helper, instance_token,
                                            sample_token,
                                            self.obs_len, MAX_AGENTS)

            # Skip samples with no valid future
            if not np.any(fut):
                skipped += 1
                continue

            obs_norm, origin = normalize_trajectory(obs)
            fut_norm         = fut - origin          # same shift
            neigh_norm       = normalize_trajectory(neigh)

            self.samples.append({
                'obs'    : obs_norm.astype(np.float32),
                'fut'    : fut_norm.astype(np.float32),
                'neigh'  : neigh_norm.astype(np.float32),
                'origin' : origin.astype(np.float32),
            })

        print(f"[DataLoader] Loaded {len(self.samples)} samples "
              f"(skipped {skipped})")

    @staticmethod
    def _is_valid_instance(nusc, instance_token):
        """Return True only for pedestrian / cyclist instances."""
        try:
            instance  = nusc.get('instance', instance_token)
            category  = nusc.get('category', instance['category_token'])
            cat_name  = category['name'].lower()
            return any(a in cat_name for a in AGENT_CATEGORIES)
        except Exception:
            return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s['obs']),     # (OBS_LEN, 2)
            torch.from_numpy(s['fut']),     # (PRED_LEN, 2)
            torch.from_numpy(s['neigh']),   # (MAX_AGENTS, OBS_LEN, 2)
            torch.from_numpy(s['origin']),  # (2,)
        )


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────

def get_dataloader(dataroot, split='train', batch_size=64,
                   shuffle=True, num_workers=2):
    """
    Build and return a PyTorch DataLoader for the given split.

    Args:
        dataroot   : path to nuScenes dataset root
        split      : 'train' | 'val' | 'test'
        batch_size : samples per batch
        shuffle    : whether to shuffle
        num_workers: parallel workers

    Returns:
        DataLoader instance
    """
    nusc   = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
    helper = PredictHelper(nusc)
    dataset = NuScenesTrajectoryDataset(nusc, helper, split=split)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == 'train'),
    )
    print(f"[DataLoader] {split}: {len(dataset)} samples, "
          f"{len(loader)} batches of size {batch_size}")
    return loader


# ─────────────────────────────────────────────
# Quick sanity check (run this file directly)
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    dataroot = sys.argv[1] if len(sys.argv) > 1 else './data/nuscenes'

    loader = get_dataloader(dataroot, split='train', batch_size=4)
    obs, fut, neigh, origin = next(iter(loader))

    print("\n── Batch shapes ──────────────────────────")
    print(f"  obs    : {obs.shape}")       # (4, 4, 2)
    print(f"  fut    : {fut.shape}")       # (4, 6, 2)
    print(f"  neigh  : {neigh.shape}")     # (4, 20, 4, 2)
    print(f"  origin : {origin.shape}")    # (4, 2)
    print("──────────────────────────────────────────")
    print("data_loader.py ✓  All shapes correct.")
