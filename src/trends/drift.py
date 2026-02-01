from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeWindow:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def _to_ts(x: Any) -> pd.Timestamp:
    ts = pd.to_datetime(x, errors="coerce", utc=True)
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {x!r}")
    return ts


def build_emerging_issues(
    trends: Dict[str, Any],
    topic_labels: Dict[int, Dict[str, Any]],
    topk: int = 10,
) -> List[Dict[str, Any]]:
    latest_vs_prev = trends.get("latest_vs_prev", {}) or {}
    if not isinstance(latest_vs_prev, dict):
        return []

    rows: List[Tuple[int, float, float]] = []
    for tid_raw, v in latest_vs_prev.items():
        if not isinstance(v, dict):
            continue
        try:
            tid = int(tid_raw)
        except Exception:
            continue

        delta = float(v.get("delta", 0.0) or 0.0)
        if delta <= 0:
            continue

        growth_rate = v.get("growth_rate", 0.0)
        try:
            growth_rate_f = float(growth_rate)
        except Exception:
            growth_rate_f = 0.0

        rows.append((tid, delta, growth_rate_f))

    rows.sort(key=lambda t: (t[1], t[2]), reverse=True)
    rows = rows[: int(topk)]

    out: List[Dict[str, Any]] = []
    for tid, delta, growth_rate in rows:
        info = topic_labels.get(int(tid), {}) if isinstance(topic_labels, dict) else {}
        label = info.get("label", f"Topic {tid}") if isinstance(info, dict) else f"Topic {tid}"
        kws = info.get("keywords", []) if isinstance(info, dict) else []
        if not isinstance(kws, list):
            kws = []
        out.append(
            {
                "topic_id": int(tid),
                "label": label,
                "topic": label,
                "delta": float(delta),
                "growth_rate": float(growth_rate),
                "keywords": kws,
            }
        )
    return out


def compute_topic_trends(
    df: pd.DataFrame,
    topic_ids: np.ndarray,
    windows: List[Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    if df is None or len(df) == 0:
        return {"window_topic_counts": {}, "latest_window": None, "prev_window": None, "latest_vs_prev": {}}

    df_local = df.copy()

    if "timestamp" not in df_local.columns:
        raise KeyError("df must contain a 'timestamp' column")

    # parse timestamp
    if not pd.api.types.is_datetime64_any_dtype(df_local["timestamp"]):
        df_local["timestamp"] = pd.to_datetime(df_local["timestamp"], errors="coerce", utc=True)

    n_before_ts = len(df_local)
    df_local = df_local.dropna(subset=["timestamp"]).copy()
    n_after_ts = len(df_local)
    dropped_ts = n_before_ts - n_after_ts
    if dropped_ts > 0:
        print(f"Trends: dropped {dropped_ts:,} rows due to invalid timestamp.")

    # align topic ids
    df_local["topic_id"] = pd.Series(topic_ids).astype(int).values

    # windows normalize
    win_objs: List[TimeWindow] = []
    for w in windows:
        win_objs.append(
            TimeWindow(
                name=str(getattr(w, "name")),
                start=_to_ts(getattr(w, "start")),
                end=_to_ts(getattr(w, "end")),
            )
        )

    if len(win_objs) == 0:
        return {"window_topic_counts": {}, "latest_window": None, "prev_window": None, "latest_vs_prev": {}}

    # counts per window
    window_topic_counts: Dict[str, Dict[int, int]] = {}
    for w in win_objs:
        mask = (df_local["timestamp"] >= w.start) & (df_local["timestamp"] <= w.end)
        counts = df_local.loc[mask, "topic_id"].value_counts().to_dict()
        window_topic_counts[w.name] = {int(k): int(v) for k, v in counts.items()}

    latest_name: str = win_objs[-1].name
    prev_name: Optional[str] = win_objs[-2].name if len(win_objs) >= 2 else None

    latest_counts = window_topic_counts.get(latest_name, {})
    prev_counts = window_topic_counts.get(prev_name, {}) if prev_name else {}

    all_topics = set(latest_counts.keys()) | set(prev_counts.keys())

    # growth_rate policy: avoid inf -> None to keep dashboards sane
    growth_cfg = (config.get("trends", {}) or {}) if isinstance(config, dict) else {}
    allow_inf = bool(growth_cfg.get("allow_infinite_growth", False))

    latest_vs_prev: Dict[int, Dict[str, Any]] = {}
    for tid in all_topics:
        l = float(latest_counts.get(tid, 0))
        p = float(prev_counts.get(tid, 0))
        delta = l - p

        if p > 0:
            growth_rate: Any = delta / p
        else:
            if l > 0 and allow_inf:
                growth_rate = float("inf")
            else:
                growth_rate = None  # safer for UI

        latest_vs_prev[int(tid)] = {"latest": l, "prev": p, "delta": delta, "growth_rate": growth_rate}

    return {
        "window_topic_counts": window_topic_counts,
        "latest_window": latest_name,
        "prev_window": prev_name,
        "latest_vs_prev": latest_vs_prev,
    }
