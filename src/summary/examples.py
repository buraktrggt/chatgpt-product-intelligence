from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / denom


def _ensure_timestamp(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce", utc=True)


def _ensure_negativity_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds '__neg' column (0..1) if no explicit negativity column exists.
    rating=1 => 1.0 (most negative), rating=5 => 0.0
    """
    out = df.copy()

    # pick an existing negativity column if present
    for c in ["negativity", "negativity_score", "avg_negativity", "neg_score", "__neg"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out["__neg"] = out[c] if c != "__neg" else out["__neg"]
            return out

    # else build proxy from rating
    if "rating" in out.columns:
        r = pd.to_numeric(out["rating"], errors="coerce")
        out["__neg"] = (5.0 - r).clip(lower=0, upper=4) / 4.0
    else:
        out["__neg"] = np.nan

    out["__neg"] = pd.to_numeric(out["__neg"], errors="coerce")
    return out


def build_topic_examples(
    df: pd.DataFrame,
    topic_ids: np.ndarray,
    embeddings: Optional[np.ndarray],
    *,
    per_topic: int = 6,
    n_evidence_representative: int = 2,
    n_evidence_most_negative: int = 2,
    n_evidence_recent: int = 2,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns:
      {
        "0": [
          {kind, text, rating, negativity, timestamp, version, version_original, version_imputed, impact},
          ...
        ],
        ...
      }

    kind values (stakeholder language):
      - "evidence_representative" : closest to centroid (if embeddings provided)
      - "evidence_most_negative"  : lowest rating / highest negativity
      - "evidence_recent"         : most recent samples
    """
    if df is None or len(df) == 0:
        return {}

    df_local = df.copy()

    tids = np.asarray(topic_ids).astype(int).ravel()
    if len(tids) != len(df_local):
        raise ValueError("topic_ids length must match df rows")

    # text column
    text_col = "text_clean" if "text_clean" in df_local.columns else ("text" if "text" in df_local.columns else None)
    if text_col is None:
        raise KeyError("df must have 'text_clean' or 'text'")

    # timestamp normalize
    if "timestamp" in df_local.columns:
        df_local["timestamp"] = _ensure_timestamp(df_local["timestamp"])
    else:
        df_local["timestamp"] = pd.NaT

    # negativity proxy
    df_local = _ensure_negativity_proxy(df_local)

    # other columns
    if "impact" not in df_local.columns:
        df_local["impact"] = np.nan
    if "version" not in df_local.columns:
        df_local["version"] = ""
    if "version_original" not in df_local.columns:
        df_local["version_original"] = df_local["version"]
    if "_version_imputed" not in df_local.columns:
        df_local["_version_imputed"] = False
    df_local["_version_imputed"] = df_local["_version_imputed"].fillna(False).astype(bool)

    # embeddings normalize (optional)
    E = None
    if embeddings is not None:
        E0 = np.asarray(embeddings)
        if E0.ndim == 2 and E0.shape[0] == len(df_local):
            E = _normalize_rows(E0)

    out: Dict[str, List[Dict[str, Any]]] = {}
    unique_tids = sorted(set(tids.tolist()))

    for tid in unique_tids:
        idx = np.where(tids == tid)[0]
        sub = df_local.iloc[idx].copy()

        rows: List[Dict[str, Any]] = []
        seen_texts = set()

        def add_row(kind: str, r: pd.Series) -> None:
            txt = _safe_str(r.get(text_col))
            if not txt:
                return
            # de-dup by exact text (cheap but effective)
            if txt in seen_texts:
                return
            seen_texts.add(txt)

            rows.append(
                {
                    "kind": kind,
                    "text": txt,
                    "rating": _safe_float(r.get("rating")),
                    "negativity": _safe_float(r.get("__neg")),
                    "timestamp": _safe_str(r.get("timestamp")),
                    "version": _safe_str(r.get("version")),
                    "version_original": _safe_str(r.get("version_original")),
                    "version_imputed": bool(r.get("_version_imputed")),
                    "impact": _safe_float(r.get("impact")),
                }
            )

        # (1) representative evidence
        if E is not None and len(idx) > 0 and n_evidence_representative > 0:
            centroid = E[idx].mean(axis=0, keepdims=True)
            centroid = _normalize_rows(centroid)[0]
            sims = (E[idx] @ centroid.reshape(-1, 1)).ravel()
            order = np.argsort(sims)[::-1]  # highest cosine sim first
            for j in order[:n_evidence_representative]:
                add_row("evidence_representative", sub.iloc[int(j)])

        # (2) most negative evidence (prefer negativity, fallback rating)
        if n_evidence_most_negative > 0:
            sub2 = sub.copy()
            sub2["__neg2"] = pd.to_numeric(sub2["__neg"], errors="coerce")
            sub2["__rating2"] = pd.to_numeric(sub2["rating"], errors="coerce")

            # sort by negativity desc, then rating asc
            sub2 = sub2.sort_values(["__neg2", "__rating2"], ascending=[False, True], na_position="last")
            for _, r in sub2.head(n_evidence_most_negative).iterrows():
                add_row("evidence_most_negative", r)

        # (3) recent evidence
        if n_evidence_recent > 0:
            sub3 = sub.sort_values("timestamp", ascending=False, na_position="last")
            for _, r in sub3.head(n_evidence_recent).iterrows():
                add_row("evidence_recent", r)

        out[str(int(tid))] = rows[:per_topic]

    return out
