# pages/04_Exports.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


@st.cache_data(show_spinner=False)
def load_summary(path_str: str) -> Dict[str, Any]:
    p = Path(path_str)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("summary.json must be a JSON object")
    return data


def _theme_aggregate(priority: pd.DataFrame) -> pd.DataFrame:
    if priority.empty:
        return priority
    if "label" not in priority.columns:
        priority["label"] = "Unlabeled"
    priority["label"] = priority["label"].fillna("").astype(str).str.strip()
    priority.loc[priority["label"] == "", "label"] = "Unlabeled"

    if "label_id" not in priority.columns:
        priority["label_id"] = priority["label"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")

    # numeric coerce
    for c in ["n_reviews"]:
        if c in priority.columns:
            priority[c] = pd.to_numeric(priority[c], errors="coerce").fillna(0).astype(int)
    for c in ["avg_rating", "priority_score"]:
        if c in priority.columns:
            priority[c] = pd.to_numeric(priority[c], errors="coerce")

    rows = []
    for lid, g in priority.groupby("label_id", dropna=False):
        n = int(g["n_reviews"].sum()) if "n_reviews" in g.columns else int(len(g))

        avg_rating = None
        if "avg_rating" in g.columns:
            avg_rating = float((g["avg_rating"].fillna(0) * g["n_reviews"]).sum() / n) if n > 0 else None

        priority_score = float(g["priority_score"].sum()) if "priority_score" in g.columns else None

        rows.append(
            {
                "label_id": str(lid),
                "label": str(g["label"].iloc[0]),
                "n_reviews": n,
                "avg_rating": avg_rating,
                "priority_score": priority_score,
                "n_clusters": int(g["topic_id"].nunique()) if "topic_id" in g.columns else int(len(g)),
            }
        )

    out = pd.DataFrame(rows).sort_values(["priority_score", "n_reviews"], ascending=False)
    return out.reset_index(drop=True)


st.set_page_config(page_title="Exports — Product Intelligence", layout="wide")
st.title("Exports")
st.caption("Stakeholder-ready artifacts: report + CSV extracts + audit.")

summary_path = st.session_state.get("summary_path")
report_path = st.session_state.get("report_path")
if not summary_path:
    st.error("summary_path missing. Go to Home.")
    st.stop()

summary = load_summary(summary_path)

priority = pd.DataFrame(_as_list(summary.get("priority_ranking", [])))
release = pd.DataFrame(_as_list(_as_dict(summary.get("release_impact", {})).get("top_versions_by_negativity", [])))
meta = _as_dict(summary.get("meta", {}))
dataset = _as_dict(summary.get("dataset", {}))

theme = _theme_aggregate(priority.copy()) if not priority.empty else pd.DataFrame()
funnel = _as_dict(meta.get("row_funnel", {}))

st.subheader("Download CSV extracts")
col1, col2, col3 = st.columns(3)

with col1:
    if not priority.empty:
        st.download_button(
            "Download priority_ranking.csv",
            data=priority.to_csv(index=False).encode("utf-8"),
            file_name="priority_ranking.csv",
            mime="text/csv",
        )
    else:
        st.info("priority_ranking empty.")

with col2:
    if not theme.empty:
        st.download_button(
            "Download theme_aggregate.csv",
            data=theme.to_csv(index=False).encode("utf-8"),
            file_name="theme_aggregate.csv",
            mime="text/csv",
        )
    else:
        st.info("theme_aggregate empty.")

with col3:
    if not release.empty:
        st.download_button(
            "Download top_versions_by_negativity.csv",
            data=release.to_csv(index=False).encode("utf-8"),
            file_name="top_versions_by_negativity.csv",
            mime="text/csv",
        )
    else:
        st.info("release impact table empty.")

st.divider()

st.subheader("Row funnel (JSON)")
if funnel:
    st.download_button(
        "Download row_funnel.json",
        data=json.dumps(funnel, indent=2).encode("utf-8"),
        file_name="row_funnel.json",
        mime="application/json",
    )
    st.json(funnel)
else:
    st.info("row_funnel missing in meta. (Update pipeline.py.)")

st.divider()

st.subheader("Report preview (Markdown)")
if report_path and Path(report_path).exists():
    md = Path(report_path).read_text(encoding="utf-8", errors="ignore")
    st.download_button(
        "Download product_health_report.md",
        data=md.encode("utf-8"),
        file_name="product_health_report.md",
        mime="text/markdown",
    )
    with st.expander("Preview", expanded=True):
        st.markdown(md)
else:
    st.warning("report.md not found. Ensure pipeline renders it into reports/exports/")

st.divider()
st.subheader("Run context")
st.json({"meta": meta, "dataset": dataset})
