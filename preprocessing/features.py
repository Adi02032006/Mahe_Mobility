"""
preprocessing/features.py
--------------------------
Feature vector per timestep (11-dim):
  [x_rel, y_rel, vx, vy, ax, ay, speed, sin_θ, cos_θ, is_pedestrian, is_cyclist]
"""
import numpy as np


# ─────────────────────────────────────────────
# Per-track feature extraction
# ─────────────────────────────────────────────

def compute_features(traj: list, vis_threshold: int = 1) -> np.ndarray:
    """
    traj: list of dicts with x, y, timestamp, visibility, is_cyclist

    Returns np.ndarray (T, 11) or None.
    Feature columns:
      0: x_abs   (kept for social pooling alignment, normalised later)
      1: y_abs
      2: vx
      3: vy
      4: ax
      5: ay
      6: speed   = sqrt(vx²+vy²)
      7: sin_θ   = vy/speed
      8: cos_θ   = vx/speed
      9: is_pedestrian (1/0)
     10: is_cyclist    (1/0)
    """
    traj = [t for t in traj if t["visibility"] >= vis_threshold]
    if len(traj) < 4:
        return None

    xs  = np.array([t["x"] for t in traj], dtype=np.float64)
    ys  = np.array([t["y"] for t in traj], dtype=np.float64)
    ts  = np.array([t["timestamp"] for t in traj], dtype=np.float64)
    is_cyc = float(traj[0]["is_cyclist"])
    is_ped = 1.0 - is_cyc

    dt = np.diff(ts) / 1e6          # µs → seconds
    dt = np.clip(dt, 1e-3, 2.0)

    vx = np.diff(xs) / dt           # len T-1
    vy = np.diff(ys) / dt

    ax = np.diff(vx) / dt[1:]       # len T-2
    ay = np.diff(vy) / dt[1:]

    # Align: drop first 2 frames
    T   = len(xs) - 2
    xs  = xs[2:].astype(np.float32)
    ys  = ys[2:].astype(np.float32)
    vx  = vx[1:].astype(np.float32)
    vy  = vy[1:].astype(np.float32)
    ax  = ax.astype(np.float32)
    ay  = ay.astype(np.float32)

    speed  = np.sqrt(vx**2 + vy**2) + 1e-6
    sin_th = vy / speed
    cos_th = vx / speed

    agent_ped = np.full(T, is_ped, dtype=np.float32)
    agent_cyc = np.full(T, is_cyc, dtype=np.float32)

    return np.stack([xs, ys, vx, vy, ax, ay, speed, sin_th, cos_th,
                     agent_ped, agent_cyc], axis=1)   # (T, 11)


# ─────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────

def normalize_window(obs: np.ndarray):
    """
    obs: (obs_len, 11)
    Translates x,y to be relative to the last observed position.
    vx,vy,ax,ay are already relative (finite differences).
    Speed, heading, agent type are rotation-invariant — keep as-is.

    Returns:
        normed : (obs_len, 11)
        origin : (2,) last observed (x, y) in global frame
    """
    origin = obs[-1, :2].copy()
    normed = obs.copy()
    normed[:, 0] -= origin[0]
    normed[:, 1] -= origin[1]
    return normed.astype(np.float32), origin.astype(np.float32)


def denormalize(pred: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """pred: (pred_len,2) or (K,pred_len,2). origin: (2,)"""
    return pred + origin


# ─────────────────────────────────────────────
# Social pooling (the key upgrade)
# ─────────────────────────────────────────────

def build_social_vector(all_feats: dict,
                         inst_token: str,
                         frame_idx: int,
                         radius: float = 15.0,
                         top_k: int = 5) -> np.ndarray:
    """
    For each neighbour within `radius` metres, compute:
        (rel_x, rel_y, rel_vx, rel_vy)
    then average-pool over the top-K nearest.

    Returns (4,) social vector. Zero-padded if no neighbours.
    This gives the model directional social context, not just distances.
    """
    if inst_token not in all_feats:
        return np.zeros(4, dtype=np.float32)

    ref = all_feats[inst_token]
    if frame_idx >= len(ref):
        return np.zeros(4, dtype=np.float32)

    rx, ry   = ref[frame_idx, 0], ref[frame_idx, 1]   # global x, y
    rvx, rvy = ref[frame_idx, 2], ref[frame_idx, 3]   # own velocity

    neighbours = []
    for tok, feats in all_feats.items():
        if tok == inst_token or frame_idx >= len(feats):
            continue
        ox, oy   = feats[frame_idx, 0], feats[frame_idx, 1]
        dist     = np.sqrt((rx - ox)**2 + (ry - oy)**2)
        if dist < radius:
            ovx, ovy = feats[frame_idx, 2], feats[frame_idx, 3]
            neighbours.append((dist, ox - rx, oy - ry, ovx - rvx, ovy - rvy))

    if not neighbours:
        return np.zeros(4, dtype=np.float32)

    neighbours.sort(key=lambda x: x[0])
    neighbours = neighbours[:top_k]

    # Average pool relative positions and velocities
    pool = np.mean([[n[1], n[2], n[3], n[4]] for n in neighbours], axis=0)
    return pool.astype(np.float32)    # (4,)
