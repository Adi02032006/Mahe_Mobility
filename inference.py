"""
inference.py
------------
Takes raw (x, y) coordinates as input and outputs 3 predicted future trajectories.

Usage:
    python inference.py

Or with custom coordinates:
    python inference.py --coords "0,0 1,0.5 2,1 3,1.5 4,2 5,2.5 6,3 7,3.5"

Input  : 8 (x, y) positions — 2 seconds of observed motion
Output : 3 predicted trajectories x 12 future positions — 3 seconds ahead
"""

import torch
import numpy as np
import argparse
from pathlib import Path

import config as cfg
from models.lstm_model import TrajectoryTransformer


def load_model(ckpt_path="checkpoints/best_model.pt"):
    device = torch.device(cfg.DEVICE)
    model = TrajectoryTransformer(
        feat_dim   = cfg.FEAT_DIM,
        hidden_dim = cfg.HIDDEN_DIM,
        social_dim = cfg.SOCIAL_DIM,
        num_heads  = cfg.NUM_HEADS,
        num_layers = cfg.NUM_ENC_LAYERS,
        ff_dim     = cfg.TF_FF_DIM,
        tf_dropout = cfg.TF_DROPOUT,
        dec_hidden = cfg.DEC_HIDDEN,
        dec_layers = cfg.DEC_LAYERS,
        pred_len   = cfg.PRED_LEN,
        K          = cfg.K,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[inference] Loaded checkpoint: epoch {ckpt['epoch']} | minADE: {ckpt['minADE']:.4f}")
    return model, device


def coords_to_features(xy, is_cyclist=False):
    """
    xy : (8, 2) array of (x, y) positions in metres
    Returns (8, 11) feature array.
    """
    xs = xy[:, 0].astype(np.float32)
    ys = xy[:, 1].astype(np.float32)
    dt = 0.5  # 4 Hz = 0.5s per frame

    vx = np.concatenate([[0.0], np.diff(xs) / dt]).astype(np.float32)
    vy = np.concatenate([[0.0], np.diff(ys) / dt]).astype(np.float32)
    ax = np.concatenate([[0.0, 0.0], np.diff(vx[1:]) / dt]).astype(np.float32)
    ay = np.concatenate([[0.0, 0.0], np.diff(vy[1:]) / dt]).astype(np.float32)

    speed  = np.sqrt(vx**2 + vy**2) + 1e-6
    sin_th = vy / speed
    cos_th = vx / speed

    is_ped = np.full(8, 0.0 if is_cyclist else 1.0, dtype=np.float32)
    is_cyc = np.full(8, 1.0 if is_cyclist else 0.0, dtype=np.float32)

    return np.stack([xs, ys, vx, vy, ax, ay, speed, sin_th, cos_th, is_ped, is_cyc], axis=1)


def predict(xy, model, device, is_cyclist=False):
    """
    xy     : (8, 2) observed positions
    Returns: (3, 12, 2) predicted future trajectories in global coords
    """
    features = coords_to_features(xy, is_cyclist)  # (8, 11)

    # Normalize — origin = last observed point
    origin = features[-1, :2].copy()
    obs_norm = features.copy()
    obs_norm[:, 0] -= origin[0]
    obs_norm[:, 1] -= origin[1]

    obs_t    = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0).to(device)
    social_t = torch.zeros(1, 4, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(obs_t, social_t)  # (1, 3, 12, 2)

    preds_np = preds.squeeze(0).cpu().numpy()  # (3, 12, 2)
    preds_np += origin  # de-normalize back to global coords
    return preds_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coords", type=str, default=None,
                        help='8 x,y pairs: "0,0 0.5,0.1 1,0.2 1.5,0.3 2,0.35 2.5,0.4 3,0.42 3.5,0.45"')
    parser.add_argument("--cyclist", action="store_true")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()

    # Default example — person walking in a straight line
    if args.coords is None:
        print("[inference] Using default example: straight-line pedestrian walk")
        xy = np.array([
            [0.0,  0.0],
            [0.5,  0.1],
            [1.0,  0.2],
            [1.5,  0.3],
            [2.0,  0.35],
            [2.5,  0.4],
            [3.0,  0.42],
            [3.5,  0.45],
        ], dtype=np.float32)
    else:
        pairs = args.coords.strip().split()
        assert len(pairs) == 8, f"Need exactly 8 coordinate pairs, got {len(pairs)}"
        xy = np.array([[float(v) for v in p.split(",")] for p in pairs], dtype=np.float32)

    if not Path(args.ckpt).exists():
        print(f"Checkpoint not found at {args.ckpt}. Run: python main.py --mode train")
        return

    model, device = load_model(args.ckpt)
    predictions = predict(xy, model, device, is_cyclist=args.cyclist)

    # Print results
    print("\n" + "="*55)
    print("INPUT — 8 observed positions (x, y) in metres:")
    print("="*55)
    for i, (x, y) in enumerate(xy):
        print(f"  t={i*0.5:.1f}s  x={x:.3f}  y={y:.3f}")

    print("\n" + "="*55)
    print("OUTPUT — 3 predicted future trajectories:")
    print("="*55)
    for k in range(predictions.shape[0]):
        print(f"\n  Mode {k+1}:")
        for step in range(predictions.shape[1]):
            x, y = predictions[k, step]
            print(f"    t=+{(step+1)*0.5:.1f}s  x={x:.3f}  y={y:.3f}")

    np.save("predictions.npy", predictions)
    print(f"\nPredictions saved to predictions.npy")
    print(f"Shape: {predictions.shape}  (3 modes, 12 steps, x/y)")


if __name__ == "__main__":
    main()