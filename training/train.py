"""
training/train.py
-----------------
Training loop with:
  • Linear warmup → cosine annealing LR schedule
  • Best-of-K loss for first WTA_START epochs
  • Winner-Takes-All loss after WTA_START (forces mode specialisation)
  • Gradient clipping
  • Best-model checkpoint (by minADE)
  • Optional TensorBoard
"""

import os, sys, time, random
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

from dataset             import get_dataloaders
from models.lstm_model   import TrajectoryTransformer, best_of_k_loss, winner_takes_all_loss
from evaluation.metrics  import evaluate_batch, aggregate_metrics


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def build_scheduler(optimizer, warmup_epochs, total_epochs, lr_min):
    """Linear warmup then cosine annealing."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine   = 0.5 * (1 + np.cos(np.pi * progress))
        floor    = lr_min / cfg.LR
        return floor + (1 - floor) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, device, use_wta: bool):
    model.train()
    total_loss = 0.0
    for batch in loader:
        obs    = batch["obs"].to(device)
        future = batch["future"].to(device)
        social = batch["social"].to(device)

        optimizer.zero_grad()
        preds = model(obs, social)

        if use_wta:
            loss = winner_takes_all_loss(preds, future)
        else:
            loss = best_of_k_loss(preds, future, cfg.DIVERSITY_W)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * obs.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_loss, all_metrics = 0.0, []
    for batch in loader:
        obs    = batch["obs"].to(device)
        future = batch["future"].to(device)
        social = batch["social"].to(device)
        preds  = model(obs, social)
        total_loss += best_of_k_loss(preds, future, 0).item() * obs.size(0)
        all_metrics.append(evaluate_batch(preds, future, cfg.MISS_THRESHOLD))
    return total_loss / len(loader.dataset), aggregate_metrics(all_metrics)


def main():
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    print(f"[train] device={device}  K={cfg.K}  epochs={cfg.NUM_EPOCHS}")

    # ── Dataset ───────────────────────────────────────────
    if not Path(cfg.DATASET_PKL).exists():
        print("[train] Building dataset from raw data…")
        from preprocessing.dataset_builder import build_dataset
        build_dataset(cfg.DATA_ROOT, cfg.DATASET_PKL,
                      cfg.OBS_LEN, cfg.PRED_LEN,
                      cfg.VIS_THRESHOLD, cfg.USE_SOCIAL,
                      cfg.SOCIAL_RADIUS, cfg.SOCIAL_TOP_K)

    train_loader, val_loader = get_dataloaders(
        cfg.DATASET_PKL, cfg.BATCH_SIZE, cfg.VAL_SPLIT, cfg.NUM_WORKERS
    )

    # ── Model ─────────────────────────────────────────────
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

    print(f"[train] params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR,
                            weight_decay=cfg.WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, cfg.LR_WARMUP,
                                 cfg.NUM_EPOCHS, cfg.LR_MIN)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(cfg.LOG_DIR); use_tb = True
    except ImportError:
        writer = None; use_tb = False

    Path(cfg.CKPT_DIR).mkdir(parents=True, exist_ok=True)
    best_ade, best_epoch = float("inf"), 0

    hdr = f"{'Ep':>4} | {'Loss':>8} | {'vLoss':>8} | {'mADE':>6} | {'mFDE':>6} | {'Miss':>5} | {'LR':>8} | Mode"
    print(f"\n{hdr}")
    print("─" * len(hdr))

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        use_wta = epoch >= cfg.WTA_START_EPOCH
        t0 = time.time()

        tr_loss = train_epoch(model, train_loader, optimizer, device, use_wta)
        vl_loss, metrics = val_epoch(model, val_loader, device)
        scheduler.step()

        ade  = metrics["minADE"]
        fde  = metrics["minFDE"]
        miss = metrics["MissRate"]
        lr   = get_lr(optimizer)
        mode = "WTA" if use_wta else "BoK"

        print(f"{epoch:>4} | {tr_loss:>8.4f} | {vl_loss:>8.4f} | "
              f"{ade:>6.4f} | {fde:>6.4f} | {miss:>5.3f} | "
              f"{lr:>8.2e} | {mode}  ({time.time()-t0:.1f}s)")

        if use_tb:
            writer.add_scalars("loss",    {"train": tr_loss, "val": vl_loss}, epoch)
            writer.add_scalars("metrics", {"minADE": ade, "minFDE": fde, "MissRate": miss}, epoch)

        if ade < best_ade:
            best_ade, best_epoch = ade, epoch
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "minADE": ade, "minFDE": fde, "MissRate": miss},
                       os.path.join(cfg.CKPT_DIR, "best_model.pt"))

        if epoch % cfg.SAVE_EVERY == 0:
            torch.save(model.state_dict(),
                       os.path.join(cfg.CKPT_DIR, f"epoch_{epoch:04d}.pt"))

    print(f"\n✓ Best minADE={best_ade:.4f} at epoch {best_epoch}")
    if use_tb: writer.close()


if __name__ == "__main__":
    main()
