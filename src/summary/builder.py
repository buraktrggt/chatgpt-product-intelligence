from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


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


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _pct(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * float(part) / float(total)


def _json_clean(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj
    if isinstance(obj, (list, tuple)):
        return [_json_clean(x) for x in obj]
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out[str(k)] = _json_clean(v)
        return out
    return str(obj)


def build_structured_summary(
    df: pd.DataFrame,
    topics: np.ndarray,
    topic_labels: Dict[int, Dict[str, Any]],
    negativity_scores: Any,  # not exported raw, but we use df["negativity"] if present
    trends: Dict[str, Any],
    release_impact: Dict[str, Any],
    config: Dict[str, Any],
    strengths: Any = None,
    embeddings: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    df_local = df.copy()
    topic_ids = np.asarray(topics).astype(int).ravel()
    if len(topic_ids) != len(df_local):
        raise ValueError("topics length must match df rows")

    rows = int(len(df_local))

    # -------------------------
    # Dataset stats
    # -------------------------
    avg_rating = None
    pct_le2 = None
    if "rating" in df_local.columns:
        rnum = pd.to_numeric(df_local["rating"], errors="coerce")
        avg_rating = _safe_float(rnum.mean())
        pct_le2 = _pct(float((rnum <= 2).sum()), float(rnum.notna().sum()))

    time_min, time_max = "", ""
    if "timestamp" in df_local.columns:
        ts = pd.to_datetime(df_local["timestamp"], errors="coerce", utc=True)
        if ts.notna().any():
            time_min = _safe_str(ts.min())
            time_max = _safe_str(ts.max())

    # -------------------------
    # Data quality
    # -------------------------
    n_missing_version = rows
    pct_missing_version = 100.0 if rows > 0 else 0.0
    if "version" in df_local.columns:
        v = df_local["version"]
        miss = v.isna() | (v.astype(str).str.strip() == "") | (v.astype(str).str.lower() == "unknown")
        n_missing_version = int(miss.sum())
        pct_missing_version = _pct(float(n_missing_version), float(len(df_local)))

    n_missing_timestamp = rows
    pct_missing_timestamp = 100.0 if rows > 0 else 0.0
    if "timestamp" in df_local.columns:
        ts = pd.to_datetime(df_local["timestamp"], errors="coerce", utc=True)
        n_missing_timestamp = int(ts.isna().sum())
        pct_missing_timestamp = _pct(float(n_missing_timestamp), float(len(df_local)))

    text_col = "text_clean" if "text_clean" in df_local.columns else ("text" if "text" in df_local.columns else None)
    n_missing_text = rows
    pct_missing_text = 100.0 if rows > 0 else 0.0
    if text_col is not None:
        t = df_local[text_col].astype(str)
        miss = t.isna() | (t.str.strip() == "") | (t.str.lower() == "nan")
        n_missing_text = int(miss.sum())
        pct_missing_text = _pct(float(n_missing_text), float(len(df_local)))

    # -------------------------
    # topics dict
    # -------------------------
    topics_out: Dict[str, Dict[str, Any]] = {}
    for tid, info in (topic_labels or {}).items():
        try:
            tid_i = int(tid)
        except Exception:
            continue

        if not isinstance(info, dict):
            topics_out[str(tid_i)] = {"label": _safe_str(info) or "General Product Issue", "keywords": []}
            continue

        lbl = _safe_str(info.get("label")).strip() or "General Product Issue"
        kws = info.get("keywords", [])
        if not isinstance(kws, list):
            kws = []
        kws = [str(k).strip() for k in kws if str(k).strip()]

        topics_out[str(tid_i)] = {**info, "label": lbl, "keywords": kws}

    # -------------------------
    # priority ranking
    # -------------------------
    df_local["__topic_id"] = topic_ids

    # Ensure negativity exists (single source of truth)
    # If pipeline wrote df["negativity"], use it; else compute fallback from rating.
    if "negativity" not in df_local.columns:
        if "rating" in df_local.columns:
            r = pd.to_numeric(df_local["rating"], errors="coerce")
            df_local["negativity"] = (5.0 - r).clip(lower=0, upper=4) / 4.0
        else:
            df_local["negativity"] = np.nan
    df_local["negativity"] = pd.to_numeric(df_local["negativity"], errors="coerce")

    pr_rows: List[Dict[str, Any]] = []

    # scoring mode
    pr_cfg = (config.get("summary", {}) or {}).get("priority_scoring", {}) if isinstance(config, dict) else {}
    mode = str(pr_cfg.get("mode", "rating_share")).strip().lower()
    # mode:
    #  - "rating_share" (current behavior)
    #  - "negativity_share" (more consistent with avg_negativity)
    #  - "hybrid" (0.5 rating + 0.5 negativity)

    for tid, sub in df_local.groupby("__topic_id"):
        tid_i = int(tid)
        n = int(len(sub))
        share = float(n) / float(max(1, rows))

        ar = None
        if "rating" in sub.columns:
            rnum = pd.to_numeric(sub["rating"], errors="coerce")
            ar = _safe_float(rnum.mean())

        an = _safe_float(sub["negativity"].mean())

        # rating component (clip 1..3 because we are negative-core; 3 is the "least bad" in core)
        ar_for_score = ar if ar is not None else 2.0
        ar_for_score = float(np.clip(float(ar_for_score), 1.0, 3.0))
        rating_component = (3.0 - ar_for_score) / 2.0  # normalize to ~[0..1]

        # negativity component already [0..1]
        neg_component = float(an) if an is not None else 0.5

        if mode == "negativity_share":
            priority_score = share * neg_component
        elif mode == "hybrid":
            priority_score = share * (0.5 * rating_component + 0.5 * neg_component)
        else:
            # legacy rating-based
            priority_score = share * (3.0 - ar_for_score)

        info = topics_out.get(str(tid_i), {})
        lbl = _safe_str(info.get("label")).strip() or f"Topic {tid_i}"
        kws = info.get("keywords", [])
        if not isinstance(kws, list):
            kws = []

        pr_rows.append(
            {
                "topic_id": tid_i,
                "label": lbl,
                "keywords": kws,
                "n_reviews": n,
                "avg_rating": ar,
                "avg_negativity": an,
                "priority_score": float(priority_score),
            }
        )

    pr_rows.sort(key=lambda r: (r.get("priority_score") or 0.0), reverse=True)

    # growth metrics
    lvp = trends.get("latest_vs_prev", {}) if isinstance(trends, dict) else {}
    if isinstance(lvp, dict):
        for r in pr_rows:
            tid_i = int(r["topic_id"])
            entry = lvp.get(tid_i) or lvp.get(str(tid_i))
            if isinstance(entry, dict):
                r["growth_delta"] = _safe_float(entry.get("delta"))
                try:
                    r["growth_rate"] = float(entry.get("growth_rate"))
                except Exception:
                    r["growth_rate"] = None
            else:
                r["growth_delta"] = None
                r["growth_rate"] = None
    else:
        for r in pr_rows:
            r["growth_delta"] = None
            r["growth_rate"] = None

    # emerging issues (optional)
    emerging: List[Dict[str, Any]] = []
    try:
        from src.trends.drift import build_emerging_issues

        tl_norm: Dict[int, Dict[str, Any]] = {}
        if isinstance(topic_labels, dict):
            for k, v in topic_labels.items():
                try:
                    tl_norm[int(k)] = v if isinstance(v, dict) else {"label": str(v), "keywords": []}
                except Exception:
                    continue

        emerging = build_emerging_issues(trends=trends, topic_labels=tl_norm, topk=10)
        emerging = _json_clean(emerging)
    except Exception:
        emerging = []

    # topic examples (optional)
    topic_examples: Dict[str, Any] = {}
    meta_err = None
    try:
        from src.summary.examples import build_topic_examples

        topic_examples = build_topic_examples(
            df=df_local,
            topic_ids=topic_ids,
            embeddings=embeddings,
            per_topic=5,
            n_representative=2,
            n_most_negative=3,
        )
        topic_examples = _json_clean(topic_examples)
    except Exception as e:
        topic_examples = {}
        meta_err = str(e)

    meta: Dict[str, Any] = {
        "project_name": (config.get("project", {}) or {}).get("name", "chatgpt-product-intelligence"),
        "windows_mode": (config.get("trends", {}) or {}).get("windows_mode", "sliding"),
        "sample_size": (config.get("data", {}) or {}).get("sample_size", None),
        "llm_enabled": bool((config.get("llm", {}) or {}).get("enabled", False)),
        "priority_scoring_mode": mode,
    }
    if meta_err:
        meta["examples_failed_reason"] = meta_err

    summary: Dict[str, Any] = {
        "meta": _json_clean(meta),
        "dataset": _json_clean(
            {
                "rows": rows,
                "avg_rating": avg_rating,
                "pct_rating_le_2": pct_le2,
                "time_min": time_min,
                "time_max": time_max,
            }
        ),
        "data_quality": _json_clean(
            {
                "n_core_rows": rows,
                "n_missing_version": n_missing_version,
                "pct_missing_version": pct_missing_version,
                "n_missing_timestamp": n_missing_timestamp,
                "pct_missing_timestamp": pct_missing_timestamp,
                "n_missing_text": n_missing_text,
                "pct_missing_text": pct_missing_text,
            }
        ),
        "topics": _json_clean(topics_out),
        "priority_ranking": _json_clean(pr_rows),
        "trends": _json_clean(trends if isinstance(trends, dict) else {}),
        "release_impact": _json_clean(release_impact if isinstance(release_impact, dict) else {}),
        "emerging_issues": _json_clean(emerging),
        "topic_examples": _json_clean(topic_examples),
    }

    return summary
