"""
preprocessing/dataset_builder.py
---------------------------------
Builds sliding-window (obs, future) pairs.
Uses upgraded features (11-dim) + social pooling (4-dim).
Saves as pickle for fast DataLoader access.
"""

import pickle
import numpy as np
from pathlib import Path

from preprocessing.extract  import extract_trajectories
from preprocessing.features import (compute_features, normalize_window,
                                    build_social_vector)

OBS_LEN  = 8
PRED_LEN = 12
STRIDE   = 1


def build_dataset(data_root: str,
                  out_path:      str   = "data/processed_dataset.pkl",
                  obs_len:       int   = OBS_LEN,
                  pred_len:      int   = PRED_LEN,
                  vis_threshold: int   = 1,
                  use_social:    bool  = True,
                  social_radius: float = 15.0,
                  social_top_k:  int   = 5):

    raw_trajs = extract_trajectories(data_root,
                                     min_track_len=obs_len + pred_len + 2)

    # Pre-compute features for ALL agents (needed for social pooling)
    all_feats = {}
    for tok, traj in raw_trajs.items():
        f = compute_features(traj, vis_threshold=vis_threshold)
        if f is not None:
            all_feats[tok] = f

    samples = []

    for inst_token, feats in all_feats.items():
        T = len(feats)
        if T < obs_len + pred_len:
            continue

        for start in range(0, T - obs_len - pred_len + 1, STRIDE):
            obs_raw = feats[start : start + obs_len]                  # (obs_len, 11)
            fut_raw = feats[start + obs_len :
                            start + obs_len + pred_len, :2]           # (pred_len, 2) — x,y only

            obs_norm, origin = normalize_window(obs_raw)
            fut_norm = fut_raw - origin                               # relative future

            # Social: at last obs frame
            frame_idx = start + obs_len - 1 + 2   # +2 offset from compute_features
            if use_social:
                social = build_social_vector(
                    all_feats, inst_token, frame_idx,
                    radius=social_radius, top_k=social_top_k
                )
            else:
                social = np.zeros(4, dtype=np.float32)

            samples.append({
                "obs":    obs_norm.astype(np.float32),   # (obs_len, 11)
                "future": fut_norm.astype(np.float32),   # (pred_len, 2)
                "origin": origin.astype(np.float32),     # (2,)
                "social": social.astype(np.float32),     # (4,)
            })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(samples, f)

    print(f"[dataset_builder] {len(samples)} samples → {out_path}")
    return samples


if __name__ == "__main__":
    import sys
    build_dataset(sys.argv[1] if len(sys.argv) > 1 else "data/v1.0-mini")
