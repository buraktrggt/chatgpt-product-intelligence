from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import numpy as np


def cluster_embeddings(
    embeddings: np.ndarray,
    config: Dict[str, Any],
    *,
    method: Optional[str] = None,
    kmeans_k: Optional[int] = None,
) -> np.ndarray:
    """
    Cluster embeddings into topic ids.

    If method/kmeans_k are provided, they override config["topics"].
    Supports:
      - kmeans (k can be integer or "elbow" via config)
      - hdbscan
    """
    topics_cfg = (config.get("topics", {}) or {})

    if method is None:
        method = topics_cfg.get("method", "kmeans")
    method = str(method).lower()

    if method == "kmeans":
        # priority: explicit arg > config
        if kmeans_k is not None:
            k = int(kmeans_k)
            return _cluster_kmeans(embeddings, k)

        # config-driven
        k_cfg = topics_cfg.get("kmeans_k", 12)

        # allow string "elbow"
        if isinstance(k_cfg, str) and k_cfg.strip().lower() == "elbow":
            elbow_cfg = (topics_cfg.get("elbow", {}) or {})
            from src.topics.elbow import choose_kmeans_k_elbow

            res = choose_kmeans_k_elbow(
                embeddings=embeddings,
                config=config,
                k_min=int(elbow_cfg.get("k_min", 6)),
                k_max=int(elbow_cfg.get("k_max", 30)),
                step=int(elbow_cfg.get("step", 2)),
                sample_n=int(elbow_cfg.get("sample_n", 30000)),
                seed=int(elbow_cfg.get("seed", 42)),
            )
            k = int(res.chosen_k)
            print(f"[ELBOW] chosen_k={k} | method={res.method} | sample_n={res.sample_n}")
            return _cluster_kmeans(embeddings, k)

        # otherwise integer
        k = int(k_cfg)
        return _cluster_kmeans(embeddings, k)

    if method == "hdbscan":
        params = topics_cfg.get("hdbscan", {}) or {}
        min_cluster_size = int(params.get("min_cluster_size", 50))
        min_samples = int(params.get("min_samples", 10))
        return _cluster_hdbscan(embeddings, min_cluster_size, min_samples)

    raise ValueError(f"Unsupported topics.method: {method}")


def _cluster_kmeans(embeddings: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    k = int(k)
    if k < 2:
        raise ValueError("kmeans k must be >= 2")

    print(f"Clustering with KMeans (k={k})")
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(np.asarray(embeddings))
    return labels.astype(int)


def _cluster_hdbscan(embeddings: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    import hdbscan

    print(f"Clustering with HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(np.asarray(embeddings))
    return labels.astype(int)
