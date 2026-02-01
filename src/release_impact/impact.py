from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


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


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


def _norm_version(s: pd.Series) -> pd.Series:
    v = s.fillna("unknown").astype(str).str.strip()
    v = v.where(v != "", "unknown")
    v = v.where(v.str.lower() != "nan", "unknown")
    v = v.where(v.str.lower() != "none", "unknown")
    v = v.where(v.str.lower() != "<na>", "unknown")
    return v


def _pick_neg_col(df: pd.DataFrame) -> str:
    for c in ["negativity", "negativity_score", "avg_negativity", "neg_score"]:
        if c in df.columns:
            return c
    return "__neg"


def _ensure_negativity(df: pd.DataFrame, neg_col: str) -> pd.DataFrame:
    out = df.copy()
    if neg_col != "__neg":
        out[neg_col] = pd.to_numeric(out[neg_col], errors="coerce")
        return out

    # approximate from rating if no negativity col exists
    if "rating" in out.columns:
        r = pd.to_numeric(out["rating"], errors="coerce")
        out["__neg"] = (5.0 - r).clip(lower=0, upper=4) / 4.0
    else:
        out["__neg"] = np.nan

    out["__neg"] = pd.to_numeric(out["__neg"], errors="coerce")
    return out


def _bucket_summary(df: pd.DataFrame, neg_col: str) -> Dict[str, Any]:
    if df is None or len(df) == 0:
        return {"n_reviews": 0, "avg_rating": None, "avg_negativity": None}

    avg_rating = None
    if "rating" in df.columns:
        avg_rating = _safe_float(pd.to_numeric(df["rating"], errors="coerce").mean())

    avg_neg = None
    if neg_col in df.columns:
        avg_neg = _safe_float(pd.to_numeric(df[neg_col], errors="coerce").mean())

    return {
        "n_reviews": int(len(df)),
        "avg_rating": avg_rating,
        "avg_negativity": avg_neg,
    }


def compute_release_impact(
    df: pd.DataFrame,
    topic_ids: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Version-level negativity summary, imputation-aware.

    Output:
      {
        "coverage": {...},
        "top_versions_by_negativity": [...],   # effective versions (can exclude unknown)
        "unknown_original_bucket": {...},      # audit: unknown BEFORE impute
        "unknown_current_bucket": {...},       # unknown AFTER impute
      }
    """
    if df is None or len(df) == 0:
        return {
            "coverage": {},
            "top_versions_by_negativity": [],
            "unknown_original_bucket": {},
            "unknown_current_bucket": {},
        }

    cfg = (config.get("release_impact", {}) or {}) if isinstance(config, dict) else {}

    min_known_rows = int(cfg.get("min_known_rows_required", 200))
    max_unknown_pct = float(cfg.get("max_unknown_pct_allowed", 60.0))
    topk = int(cfg.get("topk", 10))
    exclude_unknown_in_ranking = bool(cfg.get("exclude_unknown_in_ranking", True))

    # NEW: filter tiny versions (config already had it)
    min_reviews_per_version = int(cfg.get("min_reviews_per_version", 0))

    # NEW: show unknown row explicitly in output table (as audit row)
    include_unknown_row = bool(cfg.get("include_unknown_row_in_table", True))

    df_local = df.copy()

    # versions: effective
    if "version" not in df_local.columns:
        df_local["version"] = "unknown"
    df_local["version"] = _norm_version(df_local["version"])

    # versions: original (audit)
    has_original = "version_original" in df_local.columns
    if has_original:
        df_local["version_original"] = _norm_version(df_local["version_original"])

    # imputed flag
    if "_version_imputed" not in df_local.columns:
        df_local["_version_imputed"] = False
    else:
        df_local["_version_imputed"] = df_local["_version_imputed"].fillna(False).astype(bool)

    # negativity
    neg_col = _pick_neg_col(df_local)
    df_local = _ensure_negativity(df_local, neg_col)

    # ---- coverage (current/effective) ----
    n_rows = int(len(df_local))
    unknown_mask = df_local["version"].str.lower().eq("unknown")
    n_unknown = int(unknown_mask.sum())
    n_known = int(n_rows - n_unknown)
    pct_unknown = (100.0 * n_unknown / n_rows) if n_rows > 0 else 0.0

    # ---- audit: original unknown coverage ----
    n_unknown_original = None
    pct_unknown_original = None
    if has_original:
        unk_orig_mask = df_local["version_original"].str.lower().eq("unknown")
        n_unknown_original = int(unk_orig_mask.sum())
        pct_unknown_original = (100.0 * n_unknown_original / n_rows) if n_rows > 0 else 0.0

    # ---- impute stats ----
    n_imputed = int(df_local["_version_imputed"].sum())
    imputed_share_of_rows = (100.0 * n_imputed / n_rows) if n_rows > 0 else 0.0

    rescued_unknowns = None
    rescued_unknowns_pct = None
    if has_original:
        unk_orig_mask = df_local["version_original"].str.lower().eq("unknown")
        rescued_unknowns = int((unk_orig_mask & (~unknown_mask) & (df_local["_version_imputed"])).sum())
        denom = int(unk_orig_mask.sum())
        rescued_unknowns_pct = (100.0 * rescued_unknowns / denom) if denom > 0 else 0.0

    coverage_ok = (n_known >= min_known_rows) and (pct_unknown <= max_unknown_pct)

    coverage = {
        "n_rows": n_rows,
        "n_known_version_current": n_known,
        "n_unknown_version_current": n_unknown,
        "pct_unknown_version_current": float(pct_unknown),
        "coverage_ok": bool(coverage_ok),
        "min_known_rows_required": int(min_known_rows),
        "max_unknown_pct_allowed": float(max_unknown_pct),
        "n_imputed_rows": int(n_imputed),
        "pct_imputed_rows": float(imputed_share_of_rows),
        "min_reviews_per_version": int(min_reviews_per_version),
        "exclude_unknown_in_ranking": bool(exclude_unknown_in_ranking),
        "include_unknown_row_in_table": bool(include_unknown_row),
    }
    if has_original:
        coverage["n_unknown_version_original"] = int(n_unknown_original)
        coverage["pct_unknown_version_original"] = float(pct_unknown_original)
        coverage["rescued_unknowns"] = int(rescued_unknowns)
        coverage["rescued_unknowns_pct_of_original_unknown"] = float(rescued_unknowns_pct)

    # Buckets
    unknown_current_bucket = _bucket_summary(df_local.loc[unknown_mask].copy(), neg_col)

    unknown_original_bucket: Dict[str, Any] = {}
    if has_original:
        unk_orig_mask = df_local["version_original"].str.lower().eq("unknown")
        unknown_original_bucket = _bucket_summary(df_local.loc[unk_orig_mask].copy(), neg_col)

    # Ranking dataset
    grp = df_local.copy()
    if exclude_unknown_in_ranking:
        grp = grp.loc[~unknown_mask].copy()

    # Optional: drop tiny versions from ranking
    if min_reviews_per_version > 0 and not grp.empty:
        vc = grp["version"].value_counts()
        keep_versions = set(vc[vc >= min_reviews_per_version].index.astype(str).tolist())
        grp = grp[grp["version"].astype(str).isin(keep_versions)].copy()

    if grp.empty:
        rows_out: List[Dict[str, Any]] = []
        # still attach unknown row if requested (audit visibility)
        if include_unknown_row and unknown_current_bucket.get("n_reviews", 0) > 0:
            rows_out.append(
                {
                    "version": "unknown",
                    "n_reviews": int(unknown_current_bucket.get("n_reviews", 0)),
                    "avg_rating": unknown_current_bucket.get("avg_rating"),
                    "avg_negativity": unknown_current_bucket.get("avg_negativity"),
                    "top_topic": None,
                    "n_imputed": 0,
                    "pct_imputed": 0.0,
                    "kind": "audit_unknown_current",
                }
            )
        return {
            "coverage": coverage,
            "top_versions_by_negativity": rows_out,
            "unknown_original_bucket": unknown_original_bucket,
            "unknown_current_bucket": unknown_current_bucket,
        }

    # align topic ids to df_local index
    tids = np.asarray(topic_ids).astype(int).ravel()
    tids_ok = len(tids) == len(df_local)
    tids_series = pd.Series(tids, index=df_local.index) if tids_ok else None

    rows: List[Dict[str, Any]] = []
    for ver, g in grp.groupby("version", dropna=False):
        n = int(len(g))
        avg_rating = (
            _safe_float(pd.to_numeric(g["rating"], errors="coerce").mean())
            if "rating" in g.columns
            else None
        )
        avg_neg = _safe_float(pd.to_numeric(g[neg_col], errors="coerce").mean()) if neg_col in g.columns else None
        n_imputed_ver = int(g["_version_imputed"].sum()) if "_version_imputed" in g.columns else 0

        top_topic = None
        if tids_series is not None:
            try:
                top_topic = int(tids_series.loc[g.index].value_counts().index[0])
            except Exception:
                top_topic = None

        rows.append(
            {
                "version": str(ver),
                "n_reviews": n,
                "avg_rating": avg_rating,
                "avg_negativity": avg_neg,
                "top_topic": top_topic,
                "n_imputed": n_imputed_ver,
                "pct_imputed": (100.0 * n_imputed_ver / n) if n > 0 else 0.0,
                "kind": "ranked",
            }
        )

    def sort_key(r: Dict[str, Any]) -> tuple:
        an = _safe_float(r.get("avg_negativity"))
        an_sort = an if an is not None else -1.0
        return (an_sort, _safe_int(r.get("n_reviews"), 0))

    rows.sort(key=sort_key, reverse=True)
    rows = rows[:topk]

    # NEW: append unknown current as audit row (visible in Streamlit)
    if include_unknown_row and unknown_current_bucket.get("n_reviews", 0) > 0:
        rows.append(
            {
                "version": "unknown",
                "n_reviews": int(unknown_current_bucket.get("n_reviews", 0)),
                "avg_rating": unknown_current_bucket.get("avg_rating"),
                "avg_negativity": unknown_current_bucket.get("avg_negativity"),
                "top_topic": None,
                "n_imputed": 0,
                "pct_imputed": 0.0,
                "kind": "audit_unknown_current",
            }
        )

    return {
        "coverage": coverage,
        "top_versions_by_negativity": rows,
        "unknown_original_bucket": unknown_original_bucket,
        "unknown_current_bucket": unknown_current_bucket,
    }
