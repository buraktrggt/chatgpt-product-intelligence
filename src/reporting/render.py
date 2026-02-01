from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import math


def _is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False


def _fmt_num(x: Any, nd: int = 3) -> str:
    """
    Numeric formatter: never returns 'None'. Uses '-' for missing.
    """
    try:
        if x is None or _is_nan(x):
            return "-"
        if isinstance(x, bool):
            return str(x)
        if isinstance(x, int):
            return str(x)
        if isinstance(x, float):
            if math.isinf(x):
                return "inf"
            return f"{x:.{nd}f}"
        # allow numeric strings
        v = float(x)
        if math.isinf(v):
            return "inf"
        return f"{v:.{nd}f}"
    except Exception:
        return "-"


def _fmt_int(x: Any) -> str:
    try:
        if x is None or _is_nan(x):
            return "0"
        return str(int(float(x)))
    except Exception:
        return "0"


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _label_from_keywords(keywords: List[str], max_terms: int = 3) -> str:
    kws = [k.strip() for k in (keywords or []) if isinstance(k, str) and k.strip()]
    if not kws:
        return ""
    bigrams = [k for k in kws if " " in k]
    unigrams = [k for k in kws if " " not in k]
    picks = (bigrams + unigrams)[:max_terms]
    return " / ".join([p.title() for p in picks])


def _resolve_label(
    *,
    topic_id: Any,
    label: Any,
    keywords: Any,
    topics_dict: Dict[str, Any],
) -> str:
    """
    Never returns 'None' or empty.
    Priority:
      1) label in row if valid
      2) topics_dict[str(topic_id)]["label"] if valid
      3) build from keywords
      4) Topic {id}
    """
    # topic id
    try:
        tid = int(topic_id)
    except Exception:
        tid = -1

    # 1) row label
    lbl = _safe_str(label).strip()
    if lbl and lbl.lower() != "none":
        return lbl

    # 2) label from summary["topics"]
    node = topics_dict.get(str(tid))
    if isinstance(node, dict):
        lbl2 = _safe_str(node.get("label")).strip()
        if lbl2 and lbl2.lower() != "none":
            return lbl2
        kws2 = node.get("keywords", [])
        if isinstance(kws2, list):
            from_kws = _label_from_keywords(kws2)
            if from_kws:
                return from_kws

    # 3) from keywords in row
    kws = keywords if isinstance(keywords, list) else []
    from_kws = _label_from_keywords(kws)
    if from_kws:
        return from_kws

    # 4) fallback
    return f"Topic {tid}" if tid >= 0 else "Topic"


def render_markdown_report(summary: Dict[str, Any], config: Dict[str, Any]) -> str:
    out_dir = Path(config["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = (config.get("reporting", {}) or {}).get("filename", "product_health_report.md")
    path = out_dir / filename

    meta = summary.get("meta", {}) or {}
    ds = summary.get("dataset", {}) or {}
    pr = summary.get("priority_ranking", []) or []
    regs = summary.get("regression_alerts", []) or []
    trends = summary.get("trends", {}) or {}
    release = summary.get("release_impact", {}) or {}
    topics_dict = summary.get("topics", {}) or {}

    # latest/prev window might live under trends in your pipeline
    latest_window = meta.get("latest_window", None)
    prev_window = meta.get("prev_window", None)
    if latest_window is None and isinstance(trends, dict):
        latest_window = trends.get("latest_window", None)
    if prev_window is None and isinstance(trends, dict):
        prev_window = trends.get("prev_window", None)

    lines: List[str] = []

    project_name = meta.get("project_name", "product-intelligence")
    lines.append(f"# Product Health Report — {project_name}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"- **Scope:** Negative-driven core dataset **{_fmt_int(ds.get('rows'))}** reviews; "
        f"avg rating **{_fmt_num(ds.get('avg_rating'), 3)}**; "
        f"**{_fmt_num(ds.get('pct_rating_le_2'), 2)}%** have rating ≤ 2; positive pool kept only for future extensions."
    )
    lines.append("- **Top issue clusters by priority:**")

    # Top 8 clusters with safe labels
    for i, row in enumerate(pr[:8], start=1):
        label = _resolve_label(
            topic_id=row.get("topic_id"),
            label=row.get("label"),
            keywords=row.get("keywords"),
            topics_dict=topics_dict,
        )
        lines.append(
            f"  {i}) **{label}** (n={_fmt_int(row.get('n_reviews'))}, avg rating={_fmt_num(row.get('avg_rating'), 3)})"
        )

    # Regression alerts
    lines.append("")
    lines.append("## Regression Alerts (Latest Window Spikes)")
    lines.append("")
    if regs:
        lines.append("| rank | topic | n_reviews | avg_rating | growth_delta | growth_rate | keywords |")
        lines.append("|---:|---|---:|---:|---:|---:|---|")
        for i, r in enumerate(regs, start=1):
            label = _resolve_label(
                topic_id=r.get("topic_id"),
                label=r.get("label"),
                keywords=r.get("keywords"),
                topics_dict=topics_dict,
            )
            kw = ", ".join((r.get("keywords") or [])[:10]) if isinstance(r.get("keywords"), list) else ""
            lines.append(
                f"| {i} | {label} | {_fmt_int(r.get('n_reviews'))} | "
                f"{_fmt_num(r.get('avg_rating'), 3)} | {_fmt_num(r.get('growth_delta'), 3)} | {_fmt_num(r.get('growth_rate'), 3)} | {kw} |"
            )
    else:
        lines.append("_No regression alerts computed._")

    # Run metadata
    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append("| key | value |")
    lines.append("| --- | --- |")
    lines.append(f"| windows_mode | `{_safe_str(meta.get('windows_mode'))}` |")
    if latest_window is not None:
        lines.append(f"| latest_window | `{_safe_str(latest_window)}` |")
    if prev_window is not None:
        lines.append(f"| prev_window | `{_safe_str(prev_window)}` |")
    lines.append(f"| sample_size | `{_safe_str(meta.get('sample_size'))}` |")
    lines.append(f"| llm_enabled | `{_safe_str(meta.get('llm_enabled'))}` |")

    if meta.get("llm_skipped_reason"):
        lines.append(f"| llm_skipped_reason | `{_safe_str(meta.get('llm_skipped_reason'))}` |")
    if meta.get("llm_failed_reason"):
        lines.append(f"| llm_failed_reason | `{_safe_str(meta.get('llm_failed_reason'))}` |")

    # Dataset snapshot
    lines.append("")
    lines.append("## Dataset Snapshot")
    lines.append("")
    lines.append(f"- rows: **{_fmt_int(ds.get('rows'))}**")
    lines.append(f"- time range: **{_safe_str(ds.get('time_min'))} → {_safe_str(ds.get('time_max'))}**")
    lines.append(f"- average rating: **{_fmt_num(ds.get('avg_rating'), 3)}**")
    lines.append(f"- % rating <= 2: **{_fmt_num(ds.get('pct_rating_le_2'), 2)}%**")

    # Top priority issues table
    lines.append("")
    lines.append("## Top Priority Issues")
    lines.append("")
    lines.append("| rank | topic | n_reviews | avg_rating | growth_delta | growth_rate | priority_score | keywords |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---|")
    for i, row in enumerate(pr[:15], start=1):
        label = _resolve_label(
            topic_id=row.get("topic_id"),
            label=row.get("label"),
            keywords=row.get("keywords"),
            topics_dict=topics_dict,
        )
        kw = ", ".join((row.get("keywords") or [])[:10]) if isinstance(row.get("keywords"), list) else ""
        lines.append(
            f"| {i} | {label} | {_fmt_int(row.get('n_reviews'))} | "
            f"{_fmt_num(row.get('avg_rating'), 3)} | {_fmt_num(row.get('growth_delta'), 3)} | {_fmt_num(row.get('growth_rate'), 3)} | "
            f"{_fmt_num(row.get('priority_score'), 3)} | {kw} |"
        )

    # Emerging issues (if upstream puts them)
    lines.append("")
    lines.append("## Emerging Issues (Latest vs Previous Window)")
    lines.append("")
    emerging = None
    if isinstance(trends, dict):
        emerging = trends.get("emerging_issues", None)

    if isinstance(emerging, list) and len(emerging) > 0:
        lines.append("| rank | topic | delta | growth_rate | keywords |")
        lines.append("|---:|---|---:|---:|---|")
        for i, e in enumerate(emerging[:10], start=1):
            label = _resolve_label(
                topic_id=e.get("topic_id"),
                label=e.get("label"),
                keywords=e.get("keywords"),
                topics_dict=topics_dict,
            )
            kw = ", ".join((e.get("keywords") or [])[:10]) if isinstance(e.get("keywords"), list) else ""
            lines.append(
                f"| {i} | {label} | {_fmt_num(e.get('delta'), 3)} | {_fmt_num(e.get('growth_rate'), 3)} | {kw} |"
            )
    else:
        lines.append("_No emerging issues computed._")

    # Release impact
    lines.append("")
    lines.append("## Release Impact (Worst Versions by Negativity)")
    lines.append("")
    top_versions = release.get("top_versions_by_negativity", None)
    if isinstance(top_versions, list) and len(top_versions) > 0:
        lines.append("| rank | version | n_reviews | avg_rating | avg_negativity | top_topic |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for i, v in enumerate(top_versions[:10], start=1):
            lines.append(
                f"| {i} | {_safe_str(v.get('version'))} | {_fmt_int(v.get('n_reviews'))} | "
                f"{_fmt_num(v.get('avg_rating'), 3)} | {_fmt_num(v.get('avg_negativity'), 3)} | {_fmt_int(v.get('top_topic'))} |"
            )
    else:
        lines.append("_No release impact section computed._")

    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)
