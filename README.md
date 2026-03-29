# Intent & Trajectory Prediction
### MAHE Mobility Hackathon — Problem Statement 1
**Manipal Institute of Technology, Bengaluru**
*Centre of Excellence in Autonomous Mobility | IEEE VTS | Department of ECE*

---

## Project Overview

In an L4 urban environment, reacting to where a pedestrian **is** isn't enough — the vehicle must predict where they **will be**. This project implements a multi-modal trajectory prediction model that forecasts the future paths of pedestrians and cyclists in urban driving scenes.

Given **2 seconds of observed motion** (8 frames at 4 Hz), the model predicts **3 seconds into the future** (12 frames), outputting K=3 diverse trajectory hypotheses to account for uncertainty in human behaviour.

**Dataset:** nuScenes v1.0-mini  
**Task:** Temporal sequence prediction (regression)  
**Agents:** Pedestrians and cyclists

### Results

| Metric | Value | Description |
|--------|-------|-------------|
| **minADE** | **0.3198 m** | Mean displacement error across all predicted steps |
| **minFDE** | **0.5636 m** | Displacement error at the final predicted step |
| **MissRate** | **1.4%** | Fraction of predictions with FDE > 2 metres |

> 93.3% of predictions land within 2 metres of the actual future path.

---

## Model Architecture

```
Input (8, 11)
    │
    ▼
┌─────────────────────────┐
│   Transformer Encoder   │  — 2 layers, 2 heads, 128-dim, pre-norm
│   + Positional Encoding │  — sinusoidal, encodes temporal order
└─────────────────────────┘
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
┌──────────────┐              ┌────────────────┐
│  Enc output  │              │   Social MLP   │
│  (128-dim)   │              │   (64-dim)     │
└──────────────┘              │ avg-pool top-5 │
    │                         │ neighbours     │
    └──────────┬──────────────┘
               ▼
    ┌─────────────────────┐
    │   Context Fusion    │  — LayerNorm, 192-dim
    └─────────────────────┘
               │
     ┌─────────┼─────────┐
     ▼         ▼         ▼
 Decoder 1  Decoder 2  Decoder 3   ← K=3 independent LSTM decoders
     │         │         │            each 128-dim, 2 layers
     └─────────┴─────────┘
               │
               ▼
    Output: (B, 3, 12, 2)
    — 3 predicted trajectories
    — 12 future time steps
    — (x, y) coordinates each
```

### Feature Vector (11-dim per frame)

| Index | Feature | Description |
|-------|---------|-------------|
| 0, 1 | x, y | Relative position (origin = last observed point) |
| 2, 3 | vx, vy | Velocity via finite differencing |
| 4, 5 | ax, ay | Acceleration |
| 6 | speed | √(vx² + vy²) |
| 7 | sin θ | Heading sine |
| 8 | cos θ | Heading cosine |
| 9 | is_pedestrian | Agent type one-hot |
| 10 | is_cyclist | Agent type one-hot |

### Social Pooling

For each agent within 15m, compute `(rel_x, rel_y, rel_vx, rel_vy)` relative to the ego agent, then average-pool over the top-5 nearest neighbours → **4-dim social context vector**.

### Training Strategy

| Phase | Epochs | Loss | Description |
|-------|--------|------|-------------|
| Warmup | 1–10 | BoK + diversity | Linear LR warmup to 2e-4 |
| Best-of-K | 10–200 | BoK MSE (λ=0.001) | All 3 decoders trained jointly |
| Winner-Takes-All | 200–600 | Smooth-L1 WTA | Hard assignment, forces specialisation |

---

## Dataset

**nuScenes v1.0-mini** — public autonomous driving dataset

| Stat | Value |
|------|-------|
| Total scenes | 10 |
| Total annotations | 18,538 |
| Ped/cyclist annotations | 5,337 |
| Valid tracks (≥22 frames) | 129 |
| Training samples | 1,298 |
| Train / Val split | 85% / 15% |

Download: [nuScenes website](https://www.nuscenes.org/nuscenes#download)

After downloading, place the JSON files in:
```
data/v1.0-mini/
├── sample_annotation.json
├── instance.json
├── category.json
├── sample.json
├── scene.json
├── visibility.json
└── sample_data.json
```

---

## Project Structure

```
trajectory_prediction/
├── data/
│   └── v1.0-mini/          ← nuScenes JSON files go here
├── preprocessing/
│   ├── extract.py           ← parse nuScenes → raw trajectories
│   ├── features.py          ← 11-dim features + social pooling
│   ├── augmentation.py      ← flip, rotate, speed jitter, noise
│   └── dataset_builder.py   ← sliding window → training samples
├── models/
│   └── lstm_model.py        ← Transformer encoder + K×LSTM decoders
├── training/
│   └── train.py             ← training loop with WTA schedule
├── evaluation/
│   └── metrics.py           ← minADE, minFDE, MissRate
├── checkpoints/             ← saved model weights (auto-created)
├── dataset.py               ← PyTorch Dataset with augmentation
├── config.py                ← ALL hyperparameters in one place
└── main.py                  ← single entry point
```

---

## Setup & Installation

### Requirements

- Python 3.8+
- NVIDIA GPU (recommended) — tested on RTX 4060 with CUDA 12.4
- ~500 MB disk space for dataset + checkpoints

### Install

```bash
# Clone the repository
git clone https://github.com/Adi02032006/Mahe_Mobility.git
cd Mahe_Mobility

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy
```

### Verify Installation

```bash
python -c "import torch; import numpy; print('torch:', torch.__version__); print('GPU:', torch.cuda.is_available())"
```

Expected output:
```
torch: 2.5.1+cu124
GPU: True
```

---

## How to Run

### Full Pipeline (recommended)

```bash
python main.py
```

This automatically runs: **preprocess → train → evaluate**

### Step by Step

```bash
# Step 1: Build dataset from raw nuScenes JSONs
python main.py --mode preprocess
# Expected output: [dataset_builder] 1298 samples → data/processed_dataset.pkl

# Step 2: Train the model (600 epochs, ~35 min on RTX 4060)
python main.py --mode train
# Prints live training table with Loss, minADE, minFDE, MissRate per epoch

# Step 3: Evaluate best checkpoint
python main.py --mode evaluate
# Prints final: minADE, minFDE, MissRate
```

### Configuration

All hyperparameters are in `config.py`. Key settings:

```python
OBS_LEN        = 8      # observation window (2 seconds at 4 Hz)
PRED_LEN       = 12     # prediction horizon (3 seconds at 4 Hz)
K              = 3      # number of predicted trajectory modes
NUM_EPOCHS     = 600    # total training epochs
WTA_START_EPOCH= 200    # when Winner-Takes-All kicks in
HIDDEN_DIM     = 128    # model hidden dimension
TF_DROPOUT     = 0.3    # dropout for regularisation
```

---

## Example Outputs / Results

### Training Output (sample)

```
  Ep |     Loss |    vLoss |   mADE |   mFDE |  Miss |       LR | Mode
──────────────────────────────────────────────────────────────────────
   1 |  11.6124 |  10.1973 | 3.0328 | 5.6938 | 0.702 | 6.00e-05 | BoK
  50 |   0.4200 |   0.5100 | 0.6800 | 1.3200 | 0.210 | 2.97e-04 | BoK
 200 |   0.1050 |   0.2100 | 0.4400 | 0.8100 | 0.065 | 1.51e-04 | WTA
 440 |   0.0640 |   0.1040 | 0.3198 | 0.5636 | 0.014 | 4.03e-05 | WTA ← best
 600 |   0.0540 |   0.1214 | 0.3302 | 0.5785 | 0.019 | 1.00e-06 | WTA
```

### Final Evaluation Output

```
══ EVALUATION ══

  minADE   : 0.3198 m
  minFDE   : 0.5636 m
  MissRate : 0.014

  (checkpoint: epoch 440)
```

### What the Numbers Mean

- **minADE 0.3198 m** — on average, the best predicted path is only 32 cm from the real path
- **minFDE 0.5636 m** — at the 3-second mark, the best prediction is 56 cm from the true position
- **MissRate 1.4%** — only 1 in 70 predictions misses by more than 2 metres

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Transformer encoder | Self-attention sees all 8 obs frames simultaneously — no bottleneck vs LSTM |
| K=3 multi-modal output | Human motion is uncertain — 3 modes cover straight/left/right |
| Winner-Takes-All training | Forces each decoder to specialise — prevents mode collapse |
| 11-dim features | Explicit heading (sin/cos) avoids angle discontinuity; agent type separates pedestrian vs cyclist dynamics |
| Social pooling (vel-aware) | Passing relative velocity, not just distance — model knows where neighbours are going |
| Heavy dropout (0.3) | Critical on 1,298 samples — prevents memorisation of 10 training scenes |

---

## Hardware & Training Details

| Detail | Value |
|--------|-------|
| GPU | NVIDIA GeForce RTX 4060 Laptop |
| CUDA | 12.4 |
| PyTorch | 2.5.1+cu124 |
| Training time | ~35 minutes (600 epochs) |
| Best epoch | 440 |
| Parameters | 792,902 |

---

## Dependencies

```
torch>=2.0.0
numpy>=1.24.0
```

Install with:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy
```

---

## License

This project was developed for the MAHE Mobility Hackathon 2025.
Manipal Institute of Technology, Bengaluru — Department of Electronics & Communication Engineering.
