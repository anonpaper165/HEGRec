"""
NGCF model implementation.

Reference: Wang et al., "Neural Graph Collaborative Filtering", SIGIR 2019.

Key differences from LightGCN:
- Feature transformation matrices W1, W2 per layer
- Non-linear activation (LeakyReLU) after each layer
- Element-wise interaction term: W2 * (e ⊙ agg)
- Dropout for regularization
- Final embedding: concatenation of all layers (not mean)

Score(u,i) = user_final . item_final  (dot product in concat space)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.dropout = dropout

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        # Per-layer transformation matrices
        self.W1 = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(n_layers)])
        self.W2 = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(n_layers)])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        for w in self.W1:
            nn.init.xavier_uniform_(w.weight)
        for w in self.W2:
            nn.init.xavier_uniform_(w.weight)

    def forward(self, adj_matrix: torch.Tensor):
        """NGCF propagation with non-linear transformations.

        E^(l+1) = LeakyReLU(W1*(E^(l) + A*E^(l)) + W2*(E^(l) ⊙ A*E^(l)))

        Returns:
            user_embeddings, item_embeddings in concatenated space
        """
        E = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_layers = [E]

        for l in range(self.n_layers):
            # Graph convolution: neighborhood aggregation
            agg = torch.sparse.mm(adj_matrix, E)

            # Sum message: W1 * (self + neighbors)
            sum_emb = self.W1[l](E + agg)

            # Interaction message: W2 * (self ⊙ neighbors)
            bi_emb = self.W2[l](E * agg)

            # Non-linear activation
            E = F.leaky_relu(sum_emb + bi_emb, negative_slope=0.2)

            # Dropout
            E = F.dropout(E, p=self.dropout, training=self.training)

            # L2 normalize per node
            E = F.normalize(E, p=2, dim=1)

            all_layers.append(E)

        # Concatenation of all layers
        E_final = torch.cat(all_layers, dim=1)

        user_emb = E_final[:self.n_users]
        item_emb = E_final[self.n_users:]
        return user_emb, item_emb

    def bpr_loss(self, user_emb, item_emb, users, pos_items, neg_items):
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        reg = (
            self.user_embedding(users).norm(2).pow(2)
            + self.item_embedding(pos_items).norm(2).pow(2)
            + self.item_embedding(neg_items).norm(2).pow(2)
        ) / (2 * len(users))

        return bpr, reg
