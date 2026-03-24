"""
models/lstm_model.py
--------------------
Architecture: Transformer Encoder → Social MLP → K × LSTM Decoders

WHY this beats a plain LSTM:
─────────────────────────────
• Transformer encoder: self-attention over the obs window captures
  arbitrary long-range dependencies (e.g. a pedestrian who slowed
  down 6 frames ago, then accelerated). LSTMs see the past through
  a bottleneck hidden state; Transformers see all timesteps equally.

• Social MLP: takes average-pooled (rel_x, rel_y, rel_vx, rel_vy)
  from neighbours → tells the model WHERE others are going, not just
  how far they are.

• K=5 independent LSTM decoders: each starts from a different
  projection of the context, learning distinct motion modes
  (straight, turn left, stop, etc.).

• Winner-Takes-All (WTA) schedule: after epoch WTA_START, we hard-
  assign each sample to its closest mode and only backprop through
  that decoder. This forces specialisation and kills mode collapse.

Loss:
  L = Best-of-K MSE  +  diversity penalty
  After epoch WTA_START: L = WTA MSE (hard assignment, no diversity)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# Positional encoding for the Transformer
# ─────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        """x: (B, T, d_model)"""
        return self.dropout(x + self.pe[:, :x.size(1)])


# ─────────────────────────────────────────────────────────────────
# Transformer Encoder (replaces the LSTM encoder)
# ─────────────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int,
                 num_heads: int, num_layers: int,
                 ff_dim: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.pos_enc    = PositionalEncoding(hidden_dim, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True,    # pre-norm = more stable
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_norm    = nn.LayerNorm(hidden_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, obs_len, feat_dim)
        Returns: (B, hidden_dim) — context vector (mean-pool over time)
        """
        x = self.input_proj(obs)         # (B, T, H)
        x = self.pos_enc(x)
        x = self.transformer(x)          # (B, T, H)
        x = self.out_norm(x)
        return x.mean(dim=1)             # mean pooling over time


# ─────────────────────────────────────────────────────────────────
# Social MLP
# ─────────────────────────────────────────────────────────────────

class SocialMLP(nn.Module):
    def __init__(self, in_dim: int = 4, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, out_dim), nn.ReLU(),
        )

    def forward(self, social: torch.Tensor) -> torch.Tensor:
        return self.net(social)


# ─────────────────────────────────────────────────────────────────
# K-mode LSTM Decoder
# ─────────────────────────────────────────────────────────────────

class MultiModalDecoder(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int,
                 num_layers: int, pred_len: int, K: int, dropout: float):
        super().__init__()
        self.K        = K
        self.pred_len = pred_len
        self.hidden   = hidden_dim
        self.layers   = num_layers

        # Each mode gets its own initial hidden state projection
        self.h_proj = nn.ModuleList([
            nn.Linear(context_dim, hidden_dim * num_layers) for _ in range(K)
        ])
        self.c_proj = nn.ModuleList([
            nn.Linear(context_dim, hidden_dim * num_layers) for _ in range(K)
        ])

        # Shared LSTM body — saves params while keeping diversity via init
        self.lstm = nn.LSTM(
            input_size=2, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Per-mode output heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64), nn.GELU(),
                nn.Linear(64, 2),
            ) for _ in range(K)
        ])

    def _init_state(self, ctx: torch.Tensor, k: int):
        B = ctx.size(0)
        h = self.h_proj[k](ctx).view(B, self.layers, self.hidden).permute(1,0,2).contiguous()
        c = self.c_proj[k](ctx).view(B, self.layers, self.hidden).permute(1,0,2).contiguous()
        return h, c

    def forward(self, context: torch.Tensor, last_xy: torch.Tensor):
        """
        context : (B, context_dim)
        last_xy : (B, 2)   last observed position (relative = 0,0 after norm)
        Returns : (B, K, pred_len, 2)
        """
        B  = context.size(0)
        all_preds = []

        for k in range(self.K):
            h, c  = self._init_state(context, k)
            inp   = last_xy.unsqueeze(1)    # (B, 1, 2)
            steps = []
            for _ in range(self.pred_len):
                out, (h, c) = self.lstm(inp, (h, c))
                delta = self.heads[k](out.squeeze(1))  # (B, 2)
                steps.append(delta)
                inp = delta.unsqueeze(1)
            all_preds.append(torch.stack(steps, dim=1))   # (B, T, 2)

        return torch.stack(all_preds, dim=1)   # (B, K, T, 2)


# ─────────────────────────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────────────────────────

class TrajectoryTransformer(nn.Module):
    def __init__(self,
                 feat_dim:    int   = 11,
                 hidden_dim:  int   = 256,
                 social_dim:  int   = 64,
                 num_heads:   int   = 4,
                 num_layers:  int   = 2,
                 ff_dim:      int   = 512,
                 tf_dropout:  float = 0.1,
                 dec_hidden:  int   = 256,
                 dec_layers:  int   = 2,
                 pred_len:    int   = 12,
                 K:           int   = 5):
        super().__init__()

        self.encoder = TransformerEncoder(
            feat_dim, hidden_dim, num_heads, num_layers, ff_dim, tf_dropout
        )
        self.social = SocialMLP(in_dim=4, out_dim=social_dim)

        ctx_dim = hidden_dim + social_dim
        self.ctx_norm = nn.LayerNorm(ctx_dim)

        self.decoder = MultiModalDecoder(
            context_dim=ctx_dim, hidden_dim=dec_hidden,
            num_layers=dec_layers, pred_len=pred_len,
            K=K, dropout=tf_dropout,
        )

    def forward(self, obs: torch.Tensor, social: torch.Tensor):
        """
        obs    : (B, obs_len, feat_dim)
        social : (B, 4)
        Returns: (B, K, pred_len, 2)
        """
        enc  = self.encoder(obs)                       # (B, H)
        soc  = self.social(social)                     # (B, S)
        ctx  = self.ctx_norm(torch.cat([enc, soc], -1))
        last_xy = obs[:, -1, :2]                       # (B, 2) — already relative
        return self.decoder(ctx, last_xy)              # (B, K, T, 2)


# ─────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────

def best_of_k_loss(preds: torch.Tensor, targets: torch.Tensor,
                   diversity_weight: float = 0.1) -> torch.Tensor:
    """
    preds  : (B, K, T, 2)
    targets: (B, T, 2)

    Best-of-K: penalise only the closest mode.
    Diversity: push final-step predictions apart.
    """
    B, K, T, _ = preds.shape
    tgt_exp  = targets.unsqueeze(1).expand_as(preds)        # (B,K,T,2)
    ade_per_k = (preds - tgt_exp).norm(dim=-1).mean(dim=-1) # (B,K)

    best_idx = ade_per_k.argmin(dim=1)                       # (B,)
    idx_exp  = best_idx.view(B,1,1,1).expand(B,1,T,2)
    best_p   = preds.gather(1, idx_exp).squeeze(1)           # (B,T,2)
    bok      = F.mse_loss(best_p, targets)

    if K > 1 and diversity_weight > 0:
        finals = preds[:, :, -1, :]           # (B,K,2)
        div, cnt = 0.0, 0
        for i in range(K):
            for j in range(i+1, K):
                div += (finals[:,i,:] - finals[:,j,:]).norm(dim=-1).mean()
                cnt += 1
        return bok - diversity_weight * (div / cnt)
    return bok


def winner_takes_all_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Hard WTA: same as BoK but uses L1 for robustness.
    Gradient flows ONLY through the winning mode.
    Forces each decoder to specialise.
    """
    B, K, T, _ = preds.shape
    tgt_exp   = targets.unsqueeze(1).expand_as(preds)
    ade_per_k = (preds - tgt_exp).norm(dim=-1).mean(dim=-1)  # (B,K)
    best_idx  = ade_per_k.argmin(dim=1).detach()              # (B,) — detach!
    idx_exp   = best_idx.view(B,1,1,1).expand(B,1,T,2)
    best_p    = preds.gather(1, idx_exp).squeeze(1)
    return F.smooth_l1_loss(best_p, targets)
