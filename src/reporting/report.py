from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math


def _is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or _is_nan(x):
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _title_from_keywords(keywords: List[str], max_terms: int = 3) -> str:
    kws = [k.strip() for k in (keywords or []) if isinstance(k, str) and k.strip()]
    if not kws:
        return ""
    bigrams = [k for k in kws if " " in k]
    unigrams = [k for k in kws if " " not in k]
    picks = (bigrams + unigrams)[:max_terms]
    return " / ".join([p.title() for p in picks])


def _resolve_topic_label(
    topic_id: Any,
    label: Any,
    keywords: Any,
    topics_dict: Dict[str, Any],
) -> str:
    tid = _safe_int(topic_id, default=-1)

    lbl = _safe_str(label).strip()
    if lbl and lbl.lower() != "none":
        return lbl

    node = topics_dict.get(str(tid))
    if isinstance(node, dict):
        lbl2 = _safe_str(node.get("label")).strip()
        if lbl2 and lbl2.lower() != "none":
            return lbl2
        kws2 = node.get("keywords", [])
        if isinstance(kws2, list):
            from_kws = _title_from_keywords(kws2)
            if from_kws:
                return from_kws

    kws = keywords if isinstance(keywords, list) else []
    from_kws = _title_from_keywords(kws)
    if from_kws:
        return from_kws

    return f"Topic {tid}" if tid >= 0 else "Topic"


def _format_rating(x: Any) -> str:
    v = _safe_float(x)
    if v is None:
        return "-"
    return f"{v:.3f}"


def _format_num(x: Any, decimals: int = 3) -> str:
    v = _safe_float(x)
    if v is None:
        return "-"
    return f"{v:.{decimals}f}"


def _format_int(x: Any) -> str:
    return str(_safe_int(x, 0))


def _pick_top_issue_clusters(
    priority_ranking: List[Dict[str, Any]],
    topics_dict: Dict[str, Any],
    topn: int = 8,
) -> List[Dict[str, Any]]:
    """
    Aggregate clusters by canonical label (theme-level view).
    """

    agg: Dict[str, Dict[str, Any]] = {}

    for r in priority_ranking:
        tid = r.get("topic_id")
        label = _resolve_topic_label(
            topic_id=tid,
            label=r.get("label"),
            keywords=r.get("keywords", []),
            topics_dict=topics_dict,
        )

        n = _safe_int(r.get("n_reviews"), 0)
        avg = _safe_float(r.get("avg_rating"))

        if n <= 0:
            continue

        if label not in agg:
            agg[label] = {
                "label": label,
                "n_reviews": 0,
                "rating_sum": 0.0,
                "rating_cnt": 0,
            }

        agg[label]["n_reviews"] += n
        if avg is not None:
            agg[label]["rating_sum"] += avg * n
            agg[label]["rating_cnt"] += n

    rows = []
    for v in agg.values():
        avg_rating = (
            v["rating_sum"] / v["rating_cnt"]
            if v["rating_cnt"] > 0 else None
        )
        rows.append({
            "label": v["label"],
            "n_reviews": v["n_reviews"],
            "avg_rating": avg_rating,
        })

    rows.sort(key=lambda x: x["n_reviews"], reverse=True)
    return rows[:topn]



def _compute_regression_alerts_from_trends(
    trends: Dict[str, Any],
    priority_ranking: List[Dict[str, Any]],
    topics_dict: Dict[str, Any],
    topk: int = 5,
    min_delta: float = 2.0,
) -> List[Dict[str, Any]]:
    lvp = trends.get("latest_vs_prev", {})
    if not isinstance(lvp, dict) or not lvp:
        return []

    avg_map: Dict[int, Optional[float]] = {}
    kw_map: Dict[int, List[str]] = {}
    for r in priority_ranking:
        tid = _safe_int(r.get("topic_id"), -1)
        if tid < 0:
            continue
        avg_map[tid] = _safe_float(r.get("avg_rating"))
        kws = r.get("keywords", [])
        if isinstance(kws, list):
            kw_map[tid] = kws

    spikes = []
    for k, v in lvp.items():
        tid = _safe_int(k, -1)
        if tid < 0 or not isinstance(v, dict):
            continue
        delta = _safe_float(v.get("delta"))
        growth = _safe_float(v.get("growth_rate"))
        if delta is None:
            continue
        if delta >= float(min_delta):
            label = _resolve_topic_label(tid, None, kw_map.get(tid, []), topics_dict)
            spikes.append(
                {
                    "topic_id": tid,
                    "label": label,
                    "avg_rating": avg_map.get(tid),
                    "growth_delta": delta,
                    "growth_rate": growth,
                    "keywords": kw_map.get(tid, []),
                }
            )

    spikes.sort(key=lambda x: (_safe_float(x["growth_delta"]) or 0.0), reverse=True)
    return spikes[:topk]


def build_markdown_report(summary: Dict[str, Any], config: Dict[str, Any]) -> str:
    meta = summary.get("meta", {}) if isinstance(summary.get("meta", {}), dict) else {}
    dataset = summary.get("dataset", {}) if isinstance(summary.get("dataset", {}), dict) else {}
    trends = summary.get("trends", {}) if isinstance(summary.get("trends", {}), dict) else {}
    release = summary.get("release_impact", {}) if isinstance(summary.get("release_impact", {}), dict) else {}

    topics_dict = summary.get("topics", {})
    if not isinstance(topics_dict, dict):
        topics_dict = {}

    priority_ranking = summary.get("priority_ranking", [])
    if not isinstance(priority_ranking, list):
        priority_ranking = []

    rows = _safe_int(dataset.get("rows"), 0)
    avg_rating = _safe_float(dataset.get("avg_rating"))
    pct_le2 = _safe_float(dataset.get("pct_rating_le_2"))

    tmin = _safe_str(dataset.get("time_min"))
    tmax = _safe_str(dataset.get("time_max"))

    windows_mode = _safe_str(meta.get("windows_mode", "sliding"))
    latest_window = _safe_str(trends.get("latest_window", meta.get("latest_window", "")))
    prev_window = _safe_str(trends.get("prev_window", meta.get("prev_window", "")))
    sample_size = meta.get("sample_size", None)
    llm_enabled = bool(meta.get("llm_enabled", False))

    top_issues = _pick_top_issue_clusters(priority_ranking, topics_dict, topn=8)
    regression_alerts = _compute_regression_alerts_from_trends(
        trends=trends,
        priority_ranking=priority_ranking,
        topics_dict=topics_dict,
        topk=5,
        min_delta=2.0,
    )

    lines: List[str] = []
    lines.append("# Product Health Report — chatgpt-product-intelligence\n")

    lines.append("## Executive Summary\n")
    lines.append(
        f"- **Scope:** Negative-driven core dataset **{rows}** reviews; "
        f"avg rating **{_format_rating(avg_rating)}**; "
        f"**{_format_num(pct_le2, 2)}%** have rating ≤ 2; "
        "positive pool kept only for future extensions."
    )
    lines.append("- **Top issue clusters by priority:**")
    for i, it in enumerate(top_issues, start=1):
        lines.append(f"  {i}) **{it['label']}** (n={it['n_reviews']}, avg rating={_format_rating(it['avg_rating'])})")
    lines.append("")

    lines.append("## Regression Alerts (Latest Window Spikes)\n")
    if not regression_alerts:
        lines.append("_No regression alerts computed._\n")
    else:
        lines.append("| rank | topic | growth_delta | growth_rate | keywords |")
        lines.append("|---:|---|---:|---:|---|")
        for i, a in enumerate(regression_alerts, start=1):
            kws = a.get("keywords", [])
            if not isinstance(kws, list):
                kws = []
            lines.append(
                f"| {i} | {a['label']} | {_format_num(a.get('growth_delta'))} | "
                f"{_format_num(a.get('growth_rate'))} | {', '.join(kws[:10])} |"
            )
        lines.append("")

    lines.append("## Run Metadata\n")
    lines.append("| key | value |")
    lines.append("| --- | --- |")
    lines.append(f"| windows_mode | `{windows_mode}` |")
    if latest_window:
        lines.append(f"| latest_window | `{latest_window}` |")
    if prev_window:
        lines.append(f"| prev_window | `{prev_window}` |")
    lines.append(f"| sample_size | `{sample_size}` |")
    lines.append(f"| llm_enabled | `{str(llm_enabled)}` |")
    for k in ["llm_failed_reason", "llm_skipped_reason"]:
        if k in meta:
            lines.append(f"| {k} | `{_safe_str(meta.get(k))}` |")
    lines.append("")

    lines.append("## Dataset Snapshot\n")
    lines.append(f"- rows: **{rows}**")
    if tmin and tmax:
        lines.append(f"- time range: **{tmin} → {tmax}**")
    lines.append(f"- average rating: **{_format_rating(avg_rating)}**")
    lines.append(f"- % rating <= 2: **{_format_num(pct_le2, 2)}%**\n")

    lines.append("## Top Priority Issues\n")
    lines.append("| rank | topic | n_reviews | avg_rating | growth_delta | growth_rate | priority_score | keywords |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---|")

    lvp = trends.get("latest_vs_prev", {}) if isinstance(trends.get("latest_vs_prev", {}), dict) else {}

    for i, r in enumerate(priority_ranking[:15], start=1):
        tid = _safe_int(r.get("topic_id"), -1)
        growth_delta = None
        growth_rate = None
        entry = lvp.get(tid) if isinstance(lvp, dict) else None
        if isinstance(entry, dict):
            growth_delta = entry.get("delta")
            growth_rate = entry.get("growth_rate")

        kws = r.get("keywords", [])
        if not isinstance(kws, list):
            kws = []

        label = _resolve_topic_label(tid, r.get("label"), kws, topics_dict)

        lines.append(
            f"| {i} | {label} | {_format_int(r.get('n_reviews'))} | {_format_rating(r.get('avg_rating'))} | "
            f"{_format_num(growth_delta)} | {_format_num(growth_rate)} | {_format_num(r.get('priority_score'))} | "
            f"{', '.join(kws[:10])} |"
        )
    lines.append("")

    # ✅ FIX: emerging issues live under trends["emerging_issues"]
    lines.append("## Emerging Issues (Latest vs Previous Window)\n")
    emerging = trends.get("emerging_issues", [])
    if isinstance(emerging, list) and emerging:
        lines.append("| rank | topic | delta | growth_rate | keywords |")
        lines.append("|---:|---|---:|---:|---|")
        for i, e in enumerate(emerging[:10], start=1):
            tid = e.get("topic_id", e.get("topic"))
            kws = e.get("keywords", [])
            if not isinstance(kws, list):
                kws = []
            label = _resolve_topic_label(tid, e.get("label", e.get("topic")), kws, topics_dict)
            lines.append(
                f"| {i} | {label} | {_format_num(e.get('delta'))} | {_format_num(e.get('growth_rate'))} | {', '.join(kws[:10])} |"
            )
        lines.append("")
    else:
        lines.append("_No emerging issues computed._\n")

    lines.append("## Release Impact (Worst Versions by Negativity)\n")
    top_versions = release.get("top_versions_by_negativity", [])
    if isinstance(top_versions, list) and top_versions:
        lines.append("| rank | version | n_reviews | avg_rating | avg_negativity | top_topic |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for i, v in enumerate(top_versions[:10], start=1):
            lines.append(
                f"| {i} | {_safe_str(v.get('version'))} | {_format_int(v.get('n_reviews'))} | "
                f"{_format_rating(v.get('avg_rating'))} | {_format_num(v.get('avg_negativity'))} | "
                f"{_format_int(v.get('top_topic'))} |"
            )
        lines.append("")
    else:
        lines.append("_No release impact section computed._\n")

    return "\n".join(lines)


def build_structured_summary(*args, **kwargs):
    raise RuntimeError(
        "Do not import build_structured_summary from src.reporting.report anymore. "
        "Use build_markdown_report(summary, config) in the renderer, and keep build_structured_summary "
        "in the dedicated summary builder module."
    )
