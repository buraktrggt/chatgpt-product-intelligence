from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd


def to_json_safe(obj: Any) -> Any:
    """
    Recursively convert objects into JSON-serializable types.

    Handles:
    - numpy arrays -> lists
    - numpy scalars -> python scalars
    - pandas Timestamp -> ISO string
    - pandas NA/NaN -> None
    - sets/tuples -> lists
    """

    # None
    if obj is None:
        return None

    # Primitive
    if isinstance(obj, (str, bool, int)):
        return obj

    # float with nan/inf
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
        return obj

    # numpy scalar
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None if math.isnan(v) else ("inf" if v > 0 else "-inf")
        return v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # numpy array
    if isinstance(obj, np.ndarray):
        return [to_json_safe(x) for x in obj.tolist()]

    # pandas Timestamp
    if isinstance(obj, pd.Timestamp):
        if pd.isna(obj):
            return None
        return obj.isoformat()

    # pandas NA
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    # dict
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[str(k)] = to_json_safe(v)
        return out

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(x) for x in obj]

    # fallback (stringify)
    return str(obj)
