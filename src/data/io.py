from __future__ import annotations

import pandas as pd
from typing import Dict, Any


def load_raw_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load raw CSV and normalize columns to canonical schema.
    Canonical columns:
      - review_id, text, rating, timestamp, version, impact
    """
    raw_path = config["data"]["raw_path"]
    cols = config["data"]["columns"]

    df = pd.read_csv(raw_path)

    # Rename to canonical schema
    rename_map = {
        cols["review_id"]: "review_id",
        cols["text"]: "text",
        cols["rating"]: "rating",
        cols["timestamp"]: "timestamp",
        cols["version"]: "version",
        cols["impact"]: "impact",
    }
    df = df.rename(columns=rename_map)

    # Keep only canonical columns (ignore extras)
    keep = ["review_id", "text", "rating", "timestamp", "version", "impact"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Basic dtype normalization
    df["review_id"] = df["review_id"].astype(str)

    # rating -> numeric
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # timestamp -> datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # version -> string (can be null)
    df["version"] = df["version"].astype("string")

    # impact -> numeric (thumbs up)
    df["impact"] = pd.to_numeric(df["impact"], errors="coerce").fillna(0.0)

    # text -> string
    df["text"] = df["text"].astype("string")

    return df
