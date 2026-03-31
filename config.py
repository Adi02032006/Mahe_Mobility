"""
config.py — central config for ALL hyperparameters.
Tuned for nuScenes v1.0-mini (10 scenes, ~18k annotations).
"""
import torch

# ── Paths ────────────────────────────────────────────────
DATA_ROOT    = "data/v1.0-mini"
DATASET_PKL  = "data/processed_dataset.pkl"
CKPT_DIR     = "checkpoints"
LOG_DIR      = "logs"

# ── Preprocessing ────────────────────────────────────────
OBS_LEN        = 8        # 2 s @ 4 Hz
PRED_LEN       = 12       # 3 s @ 4 Hz
VIS_THRESHOLD  = 1        # keep all visibility levels
USE_SOCIAL     = True
SOCIAL_RADIUS  = 15.0     # metres — neighbour search radius
SOCIAL_TOP_K   = 5        # top-K neighbours to pool

# Feature dim: x,y,vx,vy,ax,ay,speed,sin_θ,cos_θ + agent_type one-hot(2) = 11
FEAT_DIM = 11

# ── Augmentation ─────────────────────────────────────────
# With only 10 scenes, augmentation gives you 4-5x effective data.
USE_AUGMENTATION  = True
AUG_FLIP_PROB     = 0.5    # horizontal flip
AUG_ROTATE_PROB   = 0.5    # random 90/180/270 rotation
AUG_SPEED_JITTER  = 0.25   # ±20% speed scale — slightly aggressive
AUG_NOISE_STD     = 0.03   # 2cm position noise

# ── Model ────────────────────────────────────────────────
# On a small dataset, a big model memorises the 10 scenes perfectly

HIDDEN_DIM     = 128      # was 256 — reduced to prevent overfitting
SOCIAL_DIM     = 64
NUM_HEADS      = 2        # was 4
NUM_ENC_LAYERS = 2
TF_FF_DIM      = 256      # was 512
TF_DROPOUT     = 0.3      # was 0.1 — more regularisation needed
DEC_HIDDEN     = 128      # was 256
DEC_LAYERS     = 2
K              = 3        # was 5 — 5 modes won't all specialise on this few samples

# ── Training ─────────────────────────────────────────────
BATCH_SIZE      = 16       # smaller batch = more gradient updates per epoch
NUM_EPOCHS      = 600
LR              = 2e-4
LR_WARMUP       = 10
LR_MIN          = 1e-6
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 1.0
DIVERSITY_W     = 0.1
WTA_START_EPOCH = 200      # slightly later — let BoK run longer on small data
VAL_SPLIT       = 0.15

# ── Eval ─────────────────────────────────────────────────
MISS_THRESHOLD  = 2.0     # metres — FDE > this = miss

# ── Hardware ─────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0            

# ── Misc ─────────────────────────────────────────────────
SEED        = 42
SAVE_EVERY  = 20
