"""Recall@K (hits / number of held-out items) on a held-out split, used for early stopping in training."""

import numpy as np
import torch


@torch.no_grad()
def recall_at_k(model, dataset, adj_matrix, split, K=20):
    model.eval()
    user_emb, item_emb = model(adj_matrix)
    recalls = []
    for u, items in split.items():
        if not items:
            continue
        scores = item_emb @ user_emb[u]
        seen = list(dataset.user_pos_items.get(u, ()))
        if seen:
            scores[seen] = -float("inf")
        top = set(torch.topk(scores, K).indices.cpu().numpy().tolist())
        recalls.append(len(top & set(items)) / len(items))
    return float(np.mean(recalls))
