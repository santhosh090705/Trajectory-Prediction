
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper

OBS_LEN    = 4
PRED_LEN   = 6
MAX_AGENTS = 20

class NuScenesTrajectoryDataset(Dataset):
    def __init__(self, nusc, helper, split="train"):
        self.helper = helper
        self.obs_len = OBS_LEN
        self.pred_len = PRED_LEN
        self.samples = []

        print(f"[DataLoader] Building dataset from nuScenes mini...")
        all_instances = nusc.instance

        skipped = 0
        for instance in all_instances:
            category = nusc.get("category", instance["category_token"])
            cat_name = category["name"].lower()
            if "pedestrian" not in cat_name and "bicycle" not in cat_name:
                skipped += 1
                continue

            ann_token = instance["first_annotation_token"]
            annotations = []
            while ann_token:
                ann = nusc.get("sample_annotation", ann_token)
                annotations.append(ann)
                ann_token = ann["next"]

            total = len(annotations)
            window = OBS_LEN + PRED_LEN

            if total < window:
                skipped += 1
                continue

            for start in range(0, total - window + 1):
                window_anns = annotations[start:start + window]
                coords = np.array(
                    [[a["translation"][0], a["translation"][1]] for a in window_anns],
                    dtype=np.float32
                )
                obs = coords[:OBS_LEN]
                fut = coords[OBS_LEN:]

                origin = obs[-1].copy()
                obs_norm = obs - origin
                fut_norm = fut - origin

                neigh = np.zeros((MAX_AGENTS, OBS_LEN, 2), dtype=np.float32)

                self.samples.append({
                    "obs":    obs_norm,
                    "fut":    fut_norm,
                    "neigh":  neigh,
                    "origin": origin,
                })

        # Split data 70/15/15
        total_samples = len(self.samples)
        train_end = int(0.70 * total_samples)
        val_end   = int(0.85 * total_samples)

        if split == "train":
            self.samples = self.samples[:train_end]
        elif split == "val":
            self.samples = self.samples[train_end:val_end]
        else:
            self.samples = self.samples[val_end:]

        print(f"[DataLoader] {split}: {len(self.samples)} samples (skipped {skipped})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["obs"]),
            torch.from_numpy(s["fut"]),
            torch.from_numpy(s["neigh"]),
            torch.from_numpy(s["origin"]),
        )


def get_dataloader(dataroot, split="train", batch_size=32,
                   shuffle=True, num_workers=0):
    nusc   = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=False)
    helper = PredictHelper(nusc)
    dataset = NuScenesTrajectoryDataset(nusc, helper, split=split)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=(split == "train"),
    )
    print(f"[DataLoader] {split}: {len(dataset)} samples, "
          f"{len(loader)} batches of size {batch_size}")
    return loader


if __name__ == "__main__":
    import sys
    dataroot = sys.argv[1] if len(sys.argv) > 1 else "./data/nuscenes"
    loader = get_dataloader(dataroot, split="train", batch_size=4)
    obs, fut, neigh, origin = next(iter(loader))
    print(f"obs: {obs.shape}, fut: {fut.shape}")
    print("data_loader.py OK")
