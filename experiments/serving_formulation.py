"""LightGCN read-outs with and without the account's embedding (variants) and Recall@20 = hits/|test| (recall_at_k)."""
from _paths import DATA, REPO, RESULTS
import json, os, sys
import numpy as np

sys.path.insert(0, f"{REPO}")
import torch
from src.dataset import RecDataset
from src.model import LightGCN

K = 20
DATASETS = [
    ("ML-100K",     "ml-100k",     4, None),
    ("ML-1M",       "ml-1m",       4, None),
    ("Gowalla",     "gowalla",     0, None),
    ("Amazon-Book", "amazon-book", 0, None),
]


def recall_at_k(user_emb, item_emb, ds, k=K):
    rec = []
    for u, ti in ds.test_dict.items():
        if not ti:
            continue
        s = item_emb @ user_emb[u]
        tr = list(ds.user_pos_items.get(u, []))
        if tr:
            s[tr] = -float("inf")
        top = torch.topk(s, k).indices.cpu().numpy()
        rec.append(len(set(int(i) for i in top) & set(ti)) / len(ti))
    return float(np.mean(rec))


def variants(model, ds, nu):
    with torch.no_grad():
        E = torch.cat([model.user_embedding.weight, model.item_embedding.weight], 0)
        layers = [E]
        for _ in range(model.n_layers):
            E = torch.sparse.mm(ds.adj_matrix, E)
            layers.append(E)
        full = torch.stack(layers, 0).mean(0)
        prop = torch.stack(layers[1:], 0).mean(0)
        agg1 = layers[1]
        return {
            "vanilla":      (full[:nu], full[nu:]),
            "drop_user_L0": (prop[:nu], full[nu:]),
            "drop_item_L0": (full[:nu], prop[nu:]),
            "drop_both_L0": (prop[:nu], prop[nu:]),
            "agg_only":     (agg1[:nu], full[nu:]),
        }
