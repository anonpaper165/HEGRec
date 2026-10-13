"""
Location padding with a k-anonymity guarantee at the upper bound (Amazon-Book and AR2023-HK).

Construction (public codebook, built from the stored histories the operator already holds):
  order the C chunks from rarest to commonest; write each user's chunk set as a bit string in that
  order; sort users by it; cut the sorted list into consecutive groups of k (the last short group
  joins the previous one); publish, per group, its first key and the union of its chunk sets
  (the cover).
Client rule (reads only the client's own chunk set and the public codebook): compute your key,
  locate the group whose key interval holds it, and send that group's cover if it contains your
  chunk set; otherwise try the next groups in key order (up to 64), else send all C chunks.
For every stored history the rule returns its own group (identical keys are never split across groups), so each sent cover is shared by at least
  k users: upper-bound linkage <= 1/k by construction (measured below, not assumed).
Stale requests (a grown history) may land in another group or fall back to all C; that is measured.

Metrics per k: upper-bound linkage (exact matching of the sent cover), smallest anonymity set,
upload relative to sparse, fraction of stale requests sending all C, and stale linkage:
  Amazon-Book: stored profile = training history, request = training + test (5,000 requests);
  AR2023-HK: stored profile = history up to the one-year-old snapshot, request = full history
  (1,000 requests, the sample convention of dataset_amazon2023_hk.py).
  Attacker: profiles whose chunk set lies inside the sent cover, ranked by Jaccard between the
  profile's own sent cover and the request's; ties split 1/k.
Output: results/location_padding2.json
"""
from _paths import DATA, RESULTS
import os, sys, json, math, time
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SLOTS = 4096
KS = (2, 5, 10, 20, 50)


def to_bits(chunk_lists, C):
    B = np.zeros((len(chunk_lists), C), dtype=bool)
    for u, cs in enumerate(chunk_lists):
        if len(cs): B[u, list(cs)] = True
    return B


class Codebook:
    def __init__(self, B, k):
        self.C = B.shape[1]
        freq = B.sum(0)
        self.order = np.argsort(freq, kind="stable")                # rarest first
        keys = self.keys(B)
        idx = np.argsort(keys, kind="stable")
        sk = keys[idx]
        starts = [0]; i = k
        while i < len(idx):                                   # never split identical keys
            while i < len(idx) and sk[i] == sk[i - 1]: i += 1
            if i < len(idx): starts.append(i)
            i += k
        if len(starts) > 1 and len(idx) - starts[-1] < k: starts.pop()
        self.bounds = np.array(starts + [len(idx)])
        self.first_key = keys[idx[self.bounds[:-1]]]
        self.covers = np.zeros((len(starts), self.C), dtype=bool)
        self.group_of = np.empty(len(idx), dtype=np.int64)
        for g in range(len(starts)):
            mem = idx[self.bounds[g]:self.bounds[g + 1]]
            self.covers[g] = B[mem].any(0); self.group_of[mem] = g

    def keys(self, B):
        packed = np.ascontiguousarray(np.packbits(B[:, self.order], axis=1))
        return packed.view(f"S{packed.shape[1]}").ravel()

    def send(self, B, probe=64):
        """returns (group index or -1 for all-C) per row."""
        keys = self.keys(B)
        g0 = np.clip(np.searchsorted(self.first_key, keys, side="right") - 1, 0, len(self.covers) - 1)
        out = np.full(len(B), -1, dtype=np.int64)
        todo = np.arange(len(B))
        for step in range(probe):
            if len(todo) == 0: break
            g = np.minimum(g0[todo] + step, len(self.covers) - 1)
            ok = ~(B[todo] & ~self.covers[g]).any(1)
            out[todo[ok]] = g[ok]; todo = todo[~ok]
        return out


class TreeCodebook:
    """A codebook whose published structure is k-anonymous (codebook_tree.py): the operator splits its users
    recursively on one chunk bit, choosing at each node the most balanced split that leaves at least k users
    on each side, until no bit does; each leaf is a group and its cover the union of its members' chunk
    sets. Published: each internal node's split bit and each leaf's cover. A client walks the tree with its
    chunk set and sends its leaf's cover, or all C chunks if the cover does not contain its set. Same
    interface as Codebook (C, covers, send)."""
    def __init__(self, B, k):
        n, self.C = B.shape
        self.bit, self.left, self.right, self.leaf = [], [], [], []      # node arrays; leaf = group id or -1
        covers = []
        def new():
            self.bit.append(-1); self.left.append(-1); self.right.append(-1); self.leaf.append(-1); return len(self.bit) - 1
        root = new(); stack = [(root, np.arange(n))]
        while stack:
            node, idx = stack.pop()
            split = -1
            if len(idx) >= 2 * k:
                s = B[idx].sum(0); ok = (s >= k) & (len(idx) - s >= k)
                if ok.any(): split = int(np.where(ok, np.abs(s - len(idx) / 2), np.inf).argmin())
            if split < 0:
                self.leaf[node] = len(covers); covers.append(B[idx].any(0)); continue
            col = B[idx, split]; a, b = new(), new()
            self.bit[node] = split; self.left[node] = a; self.right[node] = b
            stack.append((a, idx[~col])); stack.append((b, idx[col]))
        self.covers = np.array(covers, dtype=bool)
        self.bit, self.left, self.right, self.leaf = (np.array(x) for x in (self.bit, self.left, self.right, self.leaf))

    def send(self, B, probe=64):
        node = np.zeros(len(B), dtype=np.int64)
        while True:
            internal = self.leaf[node] < 0
            if not internal.any(): break
            r = np.flatnonzero(internal); bits = B[r, self.bit[node[r]]]
            node[r] = np.where(bits, self.right[node[r]], self.left[node[r]])
        g = self.leaf[node]
        ok = ~(B & ~self.covers[g]).any(1)
        return np.where(ok, g, -1)


if os.environ.get("HEGREC_CODEBOOK") == "tree":
    Codebook = TreeCodebook


def cover_bits(cb, sent):
    full = np.ones(cb.C, dtype=bool)
    return np.where((sent >= 0)[:, None], cb.covers[np.maximum(sent, 0)], full[None, :])


def reid_labels(labels):
    g = Counter(labels.tolist()); return float(np.mean([1.0 / g[x] for x in labels.tolist()]))


POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def stale_attack(ref_B, ref_cov, q_cov, q_idx):
    from _par import pmap
    rB = np.packbits(ref_B, axis=1); rC = np.packbits(ref_cov, axis=1)
    rn = POP8[rC].sum(1).astype(np.float64)
    qC = np.packbits(q_cov, axis=1)
    parts = [p for p in np.array_split(np.asarray(q_idx), max(1, 8 * int(os.environ.get("HEGREC_NPROC", 1)))) if len(p)]
    return sum(pmap(lambda p: _stale_part(rB, rC, rn, qC, p), parts)) / len(q_idx)


def _stale_part(rB, rC, rn, qC, q_idx):
    tot = 0.0
    for qi in q_idx:
        cq = qC[qi]
        feas = ~((rB & ~cq[None, :]).any(1))
        inter = POP8[rC & cq[None, :]].sum(1).astype(np.float64)
        union = rn + float(POP8[cq].sum()) - inter
        j = np.where(feas, inter / np.maximum(union, 1), -1.0)
        best = j.max()
        if j[qi] >= best: tot += 1.0 / int(np.count_nonzero(j == best))
    return tot


def run(name, ref_lists, qry_lists, C, q_idx):
    ref_B = to_bits(ref_lists, C); q_B = to_bits(qry_lists, C)
    base = float(ref_B.sum(1).mean())
    have = ref_B.any(1)
    res = {"users": int(len(ref_B)), "C": C, "mean_chunks": round(base, 3), "k": {}}
    for k in KS:
        t0 = time.time()
        cb = Codebook(ref_B[have], k)
        s_ref = np.full(len(ref_B), -1); s_ref[have] = cb.send(ref_B[have])
        s_q = cb.send(q_B)
        lab = s_ref[have]
        sizes = Counter(lab.tolist())
        ref_cov = cover_bits(cb, s_ref); q_cov = cover_bits(cb, s_q)
        rec = {"groups": int(len(cb.covers)),
               "upper_bound": round(reid_labels(lab) * have.mean(), 6),
               "min_anonymity_set": int(min(sizes.values())),
               "upload_vs_sparse": round(float(ref_cov[have].sum(1).mean()) / base, 3),
               "stale_sends_all_C_pct": round(100 * float(np.mean(s_q[q_idx] < 0)), 2),
               "stale_upload_vs_sparse": round(float(q_cov[q_idx].sum(1).mean()) / float(q_B[q_idx].sum(1).mean()), 3),
               "stale": round(stale_attack(ref_B, ref_cov, q_cov, q_idx) * (1.0 if name != "AR2023-HK" else have.mean()), 6)}
        res["k"][str(k)] = rec
        print(name, k, rec, f"[{time.time()-t0:.0f}s]", flush=True)
    return res


def main():
    out = {"note": __doc__.strip().split("\n\n")[0], "datasets": {}}
    import stale_profile as SP
    tr, te, M = SP.load("amazon-book")
    users = sorted([u for u in tr if tr[u]]); C = math.ceil(M / SLOTS)
    ref = [sorted({i // SLOTS for i in tr[u]}) for u in users]
    qry = [sorted({i // SLOTS for i in tr[u] | te.get(u, set())}) for u in users]
    q_idx = np.arange(len(users)) if os.environ.get("HEGREC_AB_FULL") else np.sort(np.random.RandomState(0).choice(len(users), 5000, replace=False))
    out["datasets"]["Amazon-Book"] = run("Amazon-Book", ref, qry, C, q_idx)
    json.dump(out, open(os.path.join(RESULTS, "location_padding2.json"), "w"), indent=2)
    if os.environ.get("HEGREC_HK_CSV"):
        import dataset_amazon2023_hk as H
        U, I, T, nU, nI = H.load_hk(); C = math.ceil(nI / SLOTS); ch = I // SLOTS
        Tcut = int(np.quantile(T, 0.90)); keep = T <= Tcut
        def lists(mask):
            key = np.unique(U[mask] * C + ch[mask]); uu = key // C; cc = key % C
            off = np.searchsorted(uu, np.arange(nU + 1))
            return [cc[off[u]:off[u + 1]].tolist() for u in range(nU)]
        full_l = lists(np.ones(len(U), dtype=bool)); ref_l = lists(keep)
        rng = np.random.default_rng(H.SEED)
        rdeg = np.bincount(U[keep], minlength=nU)
        # same sample convention as dataset_amazon2023_hk.py (its second arm, the one-year snapshot)
        have = np.flatnonzero(rdeg > 0)
        _ = rng.choice(np.flatnonzero(np.bincount(U[T <= int(np.quantile(T, 0.95))], minlength=nU) > 0), 1000, replace=False)
        q_idx = np.sort(rng.choice(have, int(os.environ.get("HEGREC_NQ", 1000)), replace=False))
        # upper bound on the full history: stored = request = full history
        ub = run("AR2023-HK-upper", full_l, full_l, C, q_idx[:10])
        st = run("AR2023-HK", ref_l, full_l, C, q_idx)
        out["datasets"]["AR2023-HK"] = {"upper_bound_run": ub, "stale_run": st}
        json.dump(out, open(os.path.join(RESULTS, "location_padding2.json"), "w"), indent=2)
    print("SAVED")


if __name__ == "__main__":
    main()
