from __future__ import annotations

from typing import Any, Dict


def maybe_augment_with_llm(summary: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    llm_cfg = (config or {}).get("llm", {}) if isinstance(config, dict) else {}
    enabled = bool(llm_cfg.get("enabled", False))

    summary.setdefault("meta", {})["llm_enabled"] = bool(enabled)
    if not enabled:
        return summary

    topics_obj = summary.get("topics", {})
    if not isinstance(topics_obj, dict) or not topics_obj:
        summary.setdefault("meta", {})["llm_skipped_reason"] = "No topics found in summary['topics']."
        return summary

    try:
        from src.topics.label import rename_topics_with_llm  # type: ignore
    except Exception as e:
        summary.setdefault("meta", {})["llm_skipped_reason"] = f"rename_topics_with_llm unavailable: {e}"
        return summary

    normalized: Dict[int, Dict[str, Any]] = {}
    for k, v in topics_obj.items():
        try:
            tid = int(k)
        except Exception:
            continue

        if isinstance(v, dict):
            info = dict(v)
        else:
            info = {"label": str(v), "keywords": []}

        lbl = info.get("label")
        if not isinstance(lbl, str) or not lbl.strip():
            info["label"] = "General Feedback"

        kws = info.get("keywords", [])
        if not isinstance(kws, list):
            info["keywords"] = []

        normalized[tid] = info

    if not normalized:
        summary.setdefault("meta", {})["llm_skipped_reason"] = "No valid integer topic ids in topics dict."
        return summary

    timeout_s = llm_cfg.get("timeout_s", None)

    try:
        if timeout_s is None:
            renamed = rename_topics_with_llm(normalized, config)  # type: ignore
        else:
            renamed = rename_topics_with_llm(normalized, config, timeout_s=timeout_s)  # type: ignore
    except Exception as e:
        summary.setdefault("meta", {})["llm_failed_reason"] = str(e)
        return summary

    out_topics: Dict[str, Dict[str, Any]] = {}
    for tid, info in renamed.items():
        if not isinstance(info, dict):
            out_topics[str(int(tid))] = {"label": "General Feedback", "keywords": []}
            continue

        lbl = info.get("label")
        if not isinstance(lbl, str) or not lbl.strip():
            lbl = "General Feedback"

        kws = info.get("keywords", [])
        if not isinstance(kws, list):
            kws = []

        out_topics[str(int(tid))] = {**info, "label": lbl, "keywords": kws}

    summary["topics"] = out_topics
    return summary
