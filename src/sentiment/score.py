from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


_DEFAULT_MAPPING = {
    1: 1.00,
    2: 0.75,
    3: 0.50,
    4: 0.25,
    5: 0.00,
}


def compute_negativity_scores(df: pd.DataFrame, config: Dict[str, Any]) -> np.ndarray:
    """
    Negativity = rating-proxy in [0, 1], where 1 = worst.

    Default mapping:
      1 -> 1.00
      2 -> 0.75
      3 -> 0.50
      4 -> 0.25
      5 -> 0.00

    Config override (optional):
      sentiment:
        negativity_from_rating:
          mapping:
            "1": 1.0
            "2": 0.8
            ...
          fillna_rating: 3.0
    """
    if df is None or len(df) == 0:
        return np.array([], dtype=float)

    cfg = (config.get("sentiment", {}) or {}).get("negativity_from_rating", {}) if isinstance(config, dict) else {}
    mapping_raw = cfg.get("mapping", None)

    mapping = dict(_DEFAULT_MAPPING)
    if isinstance(mapping_raw, dict):
        # allow keys as strings in yaml/json
        tmp = {}
        for k, v in mapping_raw.items():
            try:
                kk = int(str(k).strip())
                vv = float(v)
                tmp[kk] = vv
            except Exception:
                continue
        if tmp:
            mapping = tmp

    fillna_rating = cfg.get("fillna_rating", 3.0)
    try:
        fillna_rating = float(fillna_rating)
    except Exception:
        fillna_rating = 3.0

    rating = pd.to_numeric(df.get("rating"), errors="coerce").fillna(fillna_rating).clip(1, 5)
    neg = rating.round().astype(int).map(mapping).astype(float).to_numpy()

    # hard clip safety
    neg = np.clip(neg, 0.0, 1.0)
    return neg
