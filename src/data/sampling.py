from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def _get_sample_size(cfg: Dict[str, Any], n_rows: int) -> Optional[int]:
    data_cfg = (cfg or {}).get("data", {}) if isinstance(cfg, dict) else {}
    sample_cfg = data_cfg.get("sample", None)

    if isinstance(sample_cfg, dict):
        enabled = bool(sample_cfg.get("enabled", True))
        if not enabled:
            return None
        size = sample_cfg.get("size", None)
        if size in (None, "", "null"):
            return None
        try:
            size_i = int(size)
        except Exception:
            return None
        if size_i <= 0:
            return None
        return min(size_i, n_rows)

    size = data_cfg.get("sample_size", None)
    if size in (None, "", "null"):
        return None
    try:
        size_i = int(size)
    except Exception:
        return None
    if size_i <= 0:
        return None
    return min(size_i, n_rows)


def sample_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    n = len(df)
    k = _get_sample_size(config, n)

    if k is None or k >= n:
        print(f"Sampling: disabled (using full dataset) n={n:,}.")
        return df

    data_cfg = (config or {}).get("data", {}) if isinstance(config, dict) else {}
    sample_cfg = data_cfg.get("sample", {}) if isinstance(data_cfg.get("sample", {}), dict) else {}
    seed = sample_cfg.get("seed", data_cfg.get("seed", 42))

    try:
        seed_i = int(seed)
    except Exception:
        seed_i = 42

    rng = np.random.default_rng(seed_i)
    idx = rng.choice(n, size=k, replace=False)
    out = df.iloc[idx].copy()
    print(f"Sampling: enabled n={n:,} -> k={k:,} (seed={seed_i}).")
    return out
