"""
LightGCN training for ML-100K, ML-1M, Gowalla and Amazon-Book; writes models/{dataset}_best.pt.

Usage:
    python train_lightgcn.py --dataset gowalla
"""

import argparse
import os
import time
import torch

from src.dataset import RecDataset
from src.model import LightGCN
from src.evaluate import recall_at_k


DATASET_DEFAULTS = {
    "ml-100k":     {"epochs": 300, "lr": 1e-3, "batch_size": 2048, "reg_weight": 1e-4,
                    "emb_dim": 64, "n_layers": 3, "eval_every": 10, "rating_threshold": 4},
    "ml-1m":       {"epochs": 500, "lr": 1e-3, "batch_size": 4096, "reg_weight": 1e-4,
                    "emb_dim": 64, "n_layers": 3, "eval_every": 20, "rating_threshold": 4},
    "gowalla":     {"epochs": 500, "lr": 1e-3, "batch_size": 4096, "reg_weight": 1e-5,
                    "emb_dim": 64, "n_layers": 3, "eval_every": 20, "rating_threshold": 0},
    "amazon-book": {"epochs": 500, "lr": 1e-3, "batch_size": 4096, "reg_weight": 1e-5,
                    "emb_dim": 64, "n_layers": 3, "eval_every": 20, "rating_threshold": 0},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGCN")
    parser.add_argument("--dataset", type=str, default="ml-100k",
                        choices=list(DATASET_DEFAULTS.keys()))
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory (default: data/<dataset>)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--emb_dim", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--reg_weight", type=float, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--rating_threshold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # merge defaults with user overrides
    defaults = DATASET_DEFAULTS[args.dataset]
    for key, default_val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default_val)

    if args.data_dir is None:
        args.data_dir = f"data/{args.dataset}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = RecDataset(args.data_dir, dataset_name=args.dataset,
                         rating_threshold=args.rating_threshold, seed=args.seed)
    dataset.split_validation()
    adj = dataset.adj_matrix.to(device)

    model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_batches = max(dataset.n_train // args.batch_size, 1)
    save_path = f"models/{args.dataset}_best.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_recall, bad_evals = -1.0, 0

    print(f"\nTraining LightGCN on {args.dataset}: emb_dim={args.emb_dim}, "
          f"layers={args.n_layers}, lr={args.lr}, reg={args.reg_weight}")
    print(f"Batches per epoch: {n_batches}, batch_size={args.batch_size}")
    print(f"Save path: {save_path}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for _ in range(n_batches):
            users, pos_items, neg_items = dataset.get_train_batch(args.batch_size)
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            user_emb, item_emb = model(adj)
            bpr_loss, reg_loss = model.bpr_loss(user_emb, item_emb, users, pos_items, neg_items)
            loss = bpr_loss + args.reg_weight * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= n_batches
        elapsed = time.time() - t0

        if epoch % args.eval_every == 0:
            recall = recall_at_k(model, dataset, adj, dataset.val_dict)
            marker = ""
            if recall > best_recall:
                best_recall, bad_evals, marker = recall, 0, " *"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_recall": recall,
                    "args": vars(args),
                    "n_users": dataset.n_users,
                    "n_items": dataset.n_items,
                }, save_path)
            else:
                bad_evals += 1
            print(f"Epoch {epoch:3d} | Loss: {epoch_loss:.4f} | "
                  f"val Recall@20: {recall:.4f} | Time: {elapsed:.1f}s{marker}")
            if bad_evals >= args.patience:
                print(f"Stopping: no improvement in {args.patience} evaluations")
                break
        else:
            print(f"Epoch {epoch:3d} | Loss: {epoch_loss:.4f} | Time: {elapsed:.1f}s")

    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
