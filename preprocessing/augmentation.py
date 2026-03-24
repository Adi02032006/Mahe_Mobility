"""
preprocessing/augmentation.py
------------------------------
On-the-fly augmentation applied per sample during training.
Augmentations that preserve trajectory physics:

1. Horizontal flip     — mirrors the scene left-right
2. Random rotation     — 90 / 180 / 270 degrees (discrete, preserves scale)
3. Speed jitter        — scales all velocities by a random factor
4. Gaussian noise      — tiny positional noise to prevent overfitting

ALL augmentations operate in the RELATIVE coordinate frame
(origin already subtracted) so they don't reintroduce location bias.

Each function takes obs (obs_len, 11) + future (pred_len, 2)
and returns augmented versions of both.
"""

import numpy as np


# ─────────────────────────────────────────────
# Flip
# ─────────────────────────────────────────────

def flip_horizontal(obs: np.ndarray, future: np.ndarray):
    """
    Negate y-axis.
    obs columns: x,y,vx,vy,ax,ay,speed,sin_θ,cos_θ,ped,cyc
    Flip: y→-y, vy→-vy, ay→-ay, sin_θ→-sin_θ  (cos_θ unchanged)
    """
    obs    = obs.copy();    future = future.copy()
    obs[:, 1]  *= -1   # y
    obs[:, 3]  *= -1   # vy
    obs[:, 5]  *= -1   # ay
    obs[:, 7]  *= -1   # sin_θ
    future[:, 1] *= -1
    return obs, future


# ─────────────────────────────────────────────
# Rotation (discrete 90°)
# ─────────────────────────────────────────────

_ROT = {
    90:  np.array([[ 0, -1], [ 1,  0]], dtype=np.float32),
    180: np.array([[-1,  0], [ 0, -1]], dtype=np.float32),
    270: np.array([[ 0,  1], [-1,  0]], dtype=np.float32),
}


def rotate_90(obs: np.ndarray, future: np.ndarray, degrees: int):
    """
    degrees ∈ {90, 180, 270}
    Rotates (x,y), (vx,vy), (ax,ay) by the rotation matrix.
    Updates sin_θ, cos_θ to match new heading.
    """
    assert degrees in _ROT
    R   = _ROT[degrees]
    obs = obs.copy(); future = future.copy()

    obs[:, :2]  = obs[:, :2] @ R.T    # (x, y)
    obs[:, 2:4] = obs[:, 2:4] @ R.T   # (vx, vy)
    obs[:, 4:6] = obs[:, 4:6] @ R.T   # (ax, ay)

    # Recompute heading from new (vx, vy)
    speed = np.sqrt(obs[:, 2]**2 + obs[:, 3]**2) + 1e-6
    obs[:, 7] = obs[:, 3] / speed     # sin_θ
    obs[:, 8] = obs[:, 2] / speed     # cos_θ

    future = future @ R.T
    return obs, future


# ─────────────────────────────────────────────
# Speed jitter
# ─────────────────────────────────────────────

def speed_jitter(obs: np.ndarray, future: np.ndarray,
                 sigma: float = 0.15):
    """
    Scale all velocities (and derived future positions) by factor ~ N(1, sigma).
    Clamped to [0.6, 1.4] to stay physical.
    """
    factor = float(np.clip(np.random.normal(1.0, sigma), 0.6, 1.4))
    obs    = obs.copy(); future = future.copy()

    obs[:, 2:6] *= factor   # vx, vy, ax, ay
    obs[:, 6]   *= factor   # speed
    future       *= factor  # scale future displacements

    return obs, future


# ─────────────────────────────────────────────
# Gaussian noise
# ─────────────────────────────────────────────

def add_position_noise(obs: np.ndarray, future: np.ndarray,
                       std: float = 0.02):
    """
    Small Gaussian noise on x, y positions only.
    Simulates GPS/sensor jitter.
    """
    obs    = obs.copy(); future = future.copy()
    obs[:, :2]  += np.random.normal(0, std, obs[:, :2].shape).astype(np.float32)
    future       += np.random.normal(0, std, future.shape).astype(np.float32)
    return obs, future


# ─────────────────────────────────────────────
# Combined augmentation (called per sample in Dataset)
# ─────────────────────────────────────────────

def augment(obs: np.ndarray, future: np.ndarray,
            flip_prob:    float = 0.5,
            rotate_prob:  float = 0.5,
            speed_sigma:  float = 0.15,
            noise_std:    float = 0.02):
    """
    Apply stochastic augmentations.
    Call this only during training — not for validation/test.
    """
    # Horizontal flip
    if np.random.random() < flip_prob:
        obs, future = flip_horizontal(obs, future)

    # Random rotation
    if np.random.random() < rotate_prob:
        deg = np.random.choice([90, 180, 270])
        obs, future = rotate_90(obs, future, int(deg))

    # Speed jitter
    if speed_sigma > 0:
        obs, future = speed_jitter(obs, future, speed_sigma)

    # Position noise
    if noise_std > 0:
        obs, future = add_position_noise(obs, future, noise_std)

    return obs, future
