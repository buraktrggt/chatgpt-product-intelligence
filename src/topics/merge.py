from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class MergeResult:
    enabled: bool
    threshold: float
    merges: List[List[int]]          # groups of old topic_ids
    old_to_new: Dict[int, int]       # old topic_id -> merged topic_id
    new_topics: List[int]            # sorted unique new topic ids


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / denom


def _topic_centroids(embeddings: np.ndarray, topic_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    topic_ids = np.asarray(topic_ids).astype(int).ravel()
    E = np.asarray(embeddings)
    if E.ndim != 2 or E.shape[0] != topic_ids.shape[0]:
        raise ValueError("embeddings must be 2D and match topic_ids length")

    topics = np.array(sorted(set(topic_ids.tolist())), dtype=int)
    centroids = []
    for t in topics:
        idx = np.where(topic_ids == t)[0]
        centroids.append(E[idx].mean(axis=0))
    C = np.vstack(centroids)
    Cn = _normalize_rows(C)
    return topics, Cn


class _DSU:
    def __init__(self, items: List[int]):
        self.parent = {int(x): int(x) for x in items}
        self.rank = {int(x): 0 for x in items}

    def find(self, x: int) -> int:
        x = int(x)
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def propose_topic_merges(
    embeddings: Optional[np.ndarray],
    topic_ids: np.ndarray,
    *,
    enabled: bool = True,
    threshold: float = 0.90,
    neighbors_k: int = 8,
) -> MergeResult:
    """
    Merge proposal using centroid cosine similarity + nearest neighbors.

    Why this fixes duplicates better than your previous version:
      - previous: if too many topic pairs, it only compared a small sliding block (i+1..i+K)
        -> misses true nearest neighbors
      - now: uses NearestNeighbors to find actual nearest topics for each centroid
    """
    topics = sorted(set(np.asarray(topic_ids).astype(int).ravel().tolist()))
    if not enabled or embeddings is None:
        return MergeResult(
            enabled=False,
            threshold=float(threshold),
            merges=[],
            old_to_new={t: t for t in topics},
            new_topics=topics,
        )

    t_arr, C = _topic_centroids(embeddings, topic_ids)
    T = int(t_arr.shape[0])
    if T <= 1:
        return MergeResult(True, float(threshold), [], {t: t for t in topics}, topics)

    k = int(max(2, min(neighbors_k, T)))
    # cosine distance = 1 - cosine similarity
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(C)
    dists, idxs = nn.kneighbors(C, return_distance=True)

    # build edges sim >= threshold
    edges: List[Tuple[float, int, int]] = []
    for i in range(T):
        src_topic = int(t_arr[i])
        for jpos in range(1, k):  # skip self (jpos=0)
            j = int(idxs[i, jpos])
            dst_topic = int(t_arr[j])
            sim = 1.0 - float(dists[i, jpos])
            if sim >= float(threshold):
                a, b = (src_topic, dst_topic) if src_topic < dst_topic else (dst_topic, src_topic)
                edges.append((sim, a, b))

    # sort edges high-sim first
    edges.sort(key=lambda x: x[0], reverse=True)

    dsu = _DSU(t_arr.tolist())
    for _, a, b in edges:
        dsu.union(a, b)

    groups: Dict[int, List[int]] = {}
    for t in t_arr.tolist():
        r = dsu.find(int(t))
        groups.setdefault(r, []).append(int(t))

    merges = [sorted(v) for v in groups.values() if len(v) > 1]
    merges.sort(key=lambda g: (len(g), g), reverse=True)

    old_to_new: Dict[int, int] = {}
    for root, members in groups.items():
        new_id = int(min(members))
        for t in members:
            old_to_new[int(t)] = new_id

    new_topics = sorted(set(old_to_new.values()))
    return MergeResult(
        enabled=True,
        threshold=float(threshold),
        merges=merges,
        old_to_new=old_to_new,
        new_topics=new_topics,
    )


def apply_topic_merges(topic_ids: np.ndarray, old_to_new: Dict[int, int]) -> np.ndarray:
    arr = np.asarray(topic_ids).astype(int).ravel()
    out = np.array([int(old_to_new.get(int(t), int(t))) for t in arr], dtype=int)

    # reindex 0..T-1 for cleanliness
    uniq = sorted(set(out.tolist()))
    remap = {old: i for i, old in enumerate(uniq)}
    out = np.array([remap[int(x)] for x in out], dtype=int)
    return out
