from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / denom


@dataclass(frozen=True)
class ElbowResult:
    k_values: List[int]
    inertia: List[float]
    chosen_k: int
    method: str
    sample_n: int


def _pick_elbow_k(k_values: List[int], inertia: List[float]) -> int:
    """
    Deterministic elbow (knee) selection:
      - normalize k and inertia to [0, 1]
      - choose point with maximum distance to chord (line from first to last)
    This is stable and fast.
    """
    if not k_values:
        return 12
    if len(k_values) < 3:
        return int(k_values[-1])

    x = np.array(k_values, dtype=float)
    y = np.array(inertia, dtype=float)

    # normalize to [0,1]
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    v = p2 - p1
    v_norm = np.linalg.norm(v) + 1e-12

    dists = []
    for i in range(len(x)):
        p = np.array([x[i], y[i]])
        # distance point->line in 2D via cross product magnitude
        dist = abs(np.cross(v, p - p1)) / v_norm
        dists.append(float(dist))

    idx = int(np.argmax(dists))
    return int(k_values[idx])


def choose_kmeans_k_elbow(
    embeddings: np.ndarray,
    config: Dict[str, Any],
    *,
    k_min: int = 6,
    k_max: int = 30,
    step: int = 2,
    sample_n: int = 30000,
    seed: int = 42,
) -> ElbowResult:
    """
    Picks k for KMeans using elbow on (sampled) embeddings.

    Notes:
      - Inertia uses Euclidean. Normalizing embeddings makes Euclidean correlate with cosine similarity.
      - We sample for speed; then fit final KMeans on full embeddings with chosen k.
    """
    from sklearn.cluster import KMeans

    E = np.asarray(embeddings)
    if E.ndim != 2 or E.shape[0] < 300:
        fallback = int((config.get("topics", {}) or {}).get("kmeans_k", 12))
        return ElbowResult([], [], fallback, "fallback_small_data", int(E.shape[0]) if E.ndim == 2 else 0)

    rng = np.random.default_rng(int(seed))
    n = int(E.shape[0])
    m = int(min(max(2000, int(sample_n)), n))
    idx = rng.choice(n, size=m, replace=False)
    Es = _normalize_rows(E[idx])

    ks = list(range(int(k_min), int(k_max) + 1, int(step)))
    inertias: List[float] = []

    for k in ks:
        km = KMeans(n_clusters=int(k), random_state=int(seed), n_init="auto")
        km.fit(Es)
        inertias.append(float(km.inertia_))

    chosen = _pick_elbow_k(ks, inertias)

    return ElbowResult(
        k_values=ks,
        inertia=inertias,
        chosen_k=int(chosen),
        method="max_distance_to_chord",
        sample_n=int(m),
    )
