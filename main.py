"""
main.py — entry point.

    python main.py                    # full pipeline
    python main.py --mode preprocess
    python main.py --mode train
    python main.py --mode evaluate
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def run_preprocess():
    from preprocessing.dataset_builder import build_dataset
    print("\n══ PREPROCESSING ══")
    s = build_dataset(cfg.DATA_ROOT, cfg.DATASET_PKL,
                      cfg.OBS_LEN, cfg.PRED_LEN,
                      cfg.VIS_THRESHOLD, cfg.USE_SOCIAL,
                      cfg.SOCIAL_RADIUS, cfg.SOCIAL_TOP_K)
    print(f"Done — {len(s)} samples.")


def run_train():
    from training.train import main as train_main
    print("\n══ TRAINING ══")
    train_main()


def run_evaluate():
    import torch
    from dataset            import get_dataloaders
    from models.lstm_model  import TrajectoryTransformer
    from evaluation.metrics import evaluate_batch, aggregate_metrics

    print("\n══ EVALUATION ══")
    device    = torch.device(cfg.DEVICE)
    ckpt_path = Path(cfg.CKPT_DIR) / "best_model.pt"
    if not ckpt_path.exists():
        print("No checkpoint found. Run training first."); return

    ckpt  = torch.load(ckpt_path, map_location=device)
    model = TrajectoryTransformer(
        feat_dim=cfg.FEAT_DIM, hidden_dim=cfg.HIDDEN_DIM,
        social_dim=cfg.SOCIAL_DIM, num_heads=cfg.NUM_HEADS,
        num_layers=cfg.NUM_ENC_LAYERS, ff_dim=cfg.TF_FF_DIM,
        tf_dropout=cfg.TF_DROPOUT, dec_hidden=cfg.DEC_HIDDEN,
        dec_layers=cfg.DEC_LAYERS, pred_len=cfg.PRED_LEN, K=cfg.K,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, val_loader = get_dataloaders(cfg.DATASET_PKL, cfg.BATCH_SIZE,
                                    cfg.VAL_SPLIT, cfg.NUM_WORKERS)
    all_metrics = []
    with torch.no_grad():
        for batch in val_loader:
            preds = model(batch["obs"].to(device), batch["social"].to(device))
            all_metrics.append(evaluate_batch(preds, batch["future"].to(device),
                                              cfg.MISS_THRESHOLD))

    r = aggregate_metrics(all_metrics)
    print(f"\n  minADE   : {r['minADE']:.4f} m")
    print(f"  minFDE   : {r['minFDE']:.4f} m")
    print(f"  MissRate : {r['MissRate']:.3f}")
    print(f"\n  (checkpoint: epoch {ckpt['epoch']})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["preprocess","train","evaluate","all"],
                   default="all")
    args = p.parse_args()
    if args.mode in ("preprocess","all"): run_preprocess()
    if args.mode in ("train","all"):      run_train()
    if args.mode in ("evaluate","all"):   run_evaluate()
