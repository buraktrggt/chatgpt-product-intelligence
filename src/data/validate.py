from __future__ import annotations

import pandas as pd
from typing import Dict, Any


REQUIRED_COLS = ["review_id", "text", "rating", "timestamp", "version", "impact"]


def validate_schema(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Validate canonical schema and basic data quality constraints.
    Raises errors early if something is wrong.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required canonical columns: {missing}")

    # Basic null checks
    if df["review_id"].isna().any():
        raise ValueError("review_id contains missing values")

    if df["text"].isna().mean() > 0.05:
        # allow a small fraction of missing text
        raise ValueError("Too many missing texts in 'text' column (>5%)")

    if df["rating"].isna().mean() > 0.05:
        raise ValueError("Too many missing ratings in 'rating' column (>5%)")

    if df["timestamp"].isna().mean() > 0.05:
        raise ValueError("Too many missing timestamps in 'timestamp' column (>5%)")

    # Rating constraints (1..5 typical)
    bad_rating = df.loc[~df["rating"].between(1, 5, inclusive="both") & df["rating"].notna()]
    if len(bad_rating) > 0:
        raise ValueError(f"Found ratings outside [1,5]. Example rows: {bad_rating.head(3).to_dict(orient='records')}")

    # Impact should be non-negative
    if (df["impact"].fillna(0) < 0).any():
        raise ValueError("impact contains negative values")

    # Text basic sanity
    # (we do deeper filtering later)
    print(f"Schema OK. Rows: {len(df):,} | Columns: {list(df.columns)}")
