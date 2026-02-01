from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


def generate_embeddings(df: pd.DataFrame, config: Dict[str, Any]) -> np.ndarray:
    """
    Generate sentence embeddings for cleaned texts.
    Uses optional caching to speed up iteration.
    """
    model_name = config["embeddings"]["model_name"]
    batch_size = int(config["embeddings"].get("batch_size", 64))
    use_cache = bool(config["embeddings"].get("cache", True))

    cache_dir = Path(config["project"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache file depends on model + number of rows (MVP sampling changes this)
    cache_key = f"emb_{model_name.replace('/', '_')}_n{len(df)}.npy"
    cache_path = cache_dir / cache_key

    if use_cache and cache_path.exists():
        print(f"Loading embeddings from cache: {cache_path}")
        return np.load(cache_path)

    print(f"Generating embeddings with model: {model_name}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = df["text_clean"].fillna("").astype(str).tolist()

    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    if use_cache:
        np.save(cache_path, emb)
        print(f"Saved embeddings cache: {cache_path}")

    return emb
