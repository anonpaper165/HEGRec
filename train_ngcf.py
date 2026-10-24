"""Standard 3-layer NGCF (LeakyReLU, BPR) for ML-100K, ML-1M, Gowalla and Amazon-Book; writes models/{ds}_ngcf_3l.pt.
Usage: python train_ngcf.py --dataset gowalla
"""

import argparse
import os
import time
import torch

from src.dataset import RecDataset
from src.ngcf_model import NGCF
from src.evaluate import recall_at_k


DATASET_DEFAULTS = {
    "ml-100k":     {"epochs": 300, "lr": 1e-3, "batch_size": 2048, "reg_weight": 1e-5, "dropout": 0.1,
                    "emb_dim": 64, "eval_every": 10, "rating_threshold": 4},
    "ml-1m":       {"epochs": 500, "lr": 1e-3, "batch_size": 4096, "reg_weight": 1e-5, "dropout": 0.1,
                    "emb_dim": 64, "eval_every": 20, "rating_threshold": 4},
    "gowalla":     {"epochs": 500, "lr": 1e-3, "batch_size": 4096, "reg_weight": 1e-5, "dropout": 0.0,
                    "emb_dim": 64, "eval_every": 20, "rating_threshold": 0},
    "amazon-book": {"epochs": 500, "lr": 1e-3, "batch_size": 4096, "reg_weight": 1e-5, "dropout": 0.0,
                    "emb_dim": 64, "eval_every": 20, "rating_threshold": 0},
}


def main():
    parser = argparse.ArgumentParser(description="Train NGCF")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_DEFAULTS.keys()))
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DATASET_DEFAULTS[args.dataset]
    ds = args.dataset

    dataset = RecDataset(f"data/{ds}", dataset_name=ds, rating_threshold=cfg["rating_threshold"], seed=args.seed)
    dataset.split_validation()
    adj = dataset.adj_matrix.to(device)
    model = NGCF(n_users=dataset.n_users, n_items=dataset.n_items, emb_dim=cfg["emb_dim"],
                 n_layers=args.n_layers, dropout=cfg["dropout"]).to(device)

    save_path = f"models/{ds}_ngcf_{args.n_layers}l.pt"
    os.makedirs("models", exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    n_batches = max(dataset.n_train // cfg["batch_size"], 1)
    best_recall, bad_evals = -1.0, 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for _ in range(n_batches):
            users, pos_items, neg_items = dataset.get_train_batch(cfg["batch_size"])
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            user_emb, item_emb = model(adj)
            bpr_loss, reg_loss = model.bpr_loss(user_emb, item_emb, users, pos_items, neg_items)
            loss = bpr_loss + cfg["reg_weight"] * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= n_batches
        elapsed = time.time() - t0

        if epoch % cfg["eval_every"] == 0:
            recall = recall_at_k(model, dataset, adj, dataset.val_dict)
            marker = ""
            if recall > best_recall:
                best_recall, bad_evals, marker = recall, 0, " *"
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "model_type": "ngcf",
                            "n_layers": args.n_layers, "val_recall": recall,
                            "n_users": dataset.n_users, "n_items": dataset.n_items, "emb_dim": cfg["emb_dim"]}, save_path)
            else:
                bad_evals += 1
            print(f"  Epoch {epoch:3d} | Loss: {epoch_loss:.4f} | val R@20: {recall:.4f} | {elapsed:.1f}s{marker}")
            if bad_evals >= args.patience:
                print(f"  Stopping: no improvement in {args.patience} evaluations")
                break

    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
