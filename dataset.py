"""
dataset.py
----------
PyTorch Dataset with on-the-fly augmentation for training split.
"""

import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from preprocessing.augmentation import augment
import config as cfg


class TrajectoryDataset(Dataset):
    def __init__(self, pkl_path: str, training: bool = False):
        with open(pkl_path, "rb") as f:
            self.samples  = pickle.load(f)
        self.training = training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        obs = s["obs"].copy()       # (obs_len, 11)
        fut = s["future"].copy()    # (pred_len, 2)

        if self.training and cfg.USE_AUGMENTATION:
            obs, fut = augment(
                obs, fut,
                flip_prob   = cfg.AUG_FLIP_PROB,
                rotate_prob = cfg.AUG_ROTATE_PROB,
                speed_sigma = cfg.AUG_SPEED_JITTER,
                noise_std   = cfg.AUG_NOISE_STD,
            )

        return {
            "obs":    torch.tensor(obs,        dtype=torch.float32),
            "future": torch.tensor(fut,        dtype=torch.float32),
            "origin": torch.tensor(s["origin"],dtype=torch.float32),
            "social": torch.tensor(s["social"],dtype=torch.float32),
        }


def get_dataloaders(pkl_path: str,
                    batch_size:  int   = 64,
                    val_split:   float = 0.15,
                    num_workers: int   = 0):

    full_ds = TrajectoryDataset(pkl_path, training=False)
    n       = len(full_ds)
    n_val   = max(1, int(n * val_split))
    n_train = n - n_val

    g = torch.Generator().manual_seed(42)
    train_idx, val_idx = random_split(range(n), [n_train, n_val], generator=g)

    # Training dataset has augmentation ON
    train_ds = TrajectoryDataset(pkl_path, training=True)
    train_ds.samples = [full_ds.samples[i] for i in train_idx.indices]

    val_ds = TrajectoryDataset(pkl_path, training=False)
    val_ds.samples = [full_ds.samples[i] for i in val_idx.indices]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"[dataset] train={len(train_ds)}  val={len(val_ds)}  "
          f"(aug={'ON' if cfg.USE_AUGMENTATION else 'OFF'})")
    return train_loader, val_loader
