# pages/01_Executive.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
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


def _fmt_float(x: Any, nd: int = 3) -> str:
    v = _safe_float(x)
    if v is None:
        return "-"
    return f"{v:.{nd}f}"


def _fmt_pct(x: Any, nd: int = 2) -> str:
    v = _safe_float(x)
    if v is None:
        return "-"
    return f"{v:.{nd}f}%"


@st.cache_data(show_spinner=False)
def load_summary(path_str: str) -> Dict[str, Any]:
    p = Path(path_str)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("summary.json must be a JSON object")
    return data


def priority_df(summary: Dict[str, Any]) -> pd.DataFrame:
    pr = _as_list(summary.get("priority_ranking", []))
    df = pd.DataFrame(pr)
    if df.empty:
        return df

    for c in ["topic_id", "n_reviews"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: _safe_int(x, 0))

    for c in ["avg_rating", "growth_delta", "growth_rate", "priority_score"]:
        if c in df.columns:
            df[c] = df[c].apply(_safe_float)

    # keywords
    if "keywords" in df.columns:
        df["keywords"] = df["keywords"].apply(lambda x: x if isinstance(x, list) else [])
        df["keywords_str"] = df["keywords"].apply(lambda xs: ", ".join([str(x) for x in xs[:12]]))
    else:
        df["keywords_str"] = ""

    if "label" not in df.columns:
        df["label"] = ""
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    df.loc[df["label"] == "", "label"] = "Unlabeled"

    # optional canonical
    if "label_id" not in df.columns:
        df["label_id"] = df["label"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")

    return df


def aggregate_theme(df_topics: pd.DataFrame) -> pd.DataFrame:
    if df_topics.empty:
        return df_topics

    def wavg(g: pd.DataFrame, col: str) -> Optional[float]:
        if col not in g.columns:
            return None
        n = g["n_reviews"].sum()
        if n <= 0:
            return None
        vals = g[col]
        if vals.isna().all():
            return None
        return float((vals.fillna(0) * g["n_reviews"]).sum() / n)

    def merge_keywords(g: pd.DataFrame) -> str:
        seen = set()
        out: List[str] = []
        g = g.sort_values("n_reviews", ascending=False)
        for _, r in g.iterrows():
            kws = r.get("keywords", [])
            if not isinstance(kws, list):
                continue
            for k in kws:
                k = str(k).strip()
                if not k or k in seen:
                    continue
                seen.add(k)
                out.append(k)
                if len(out) >= 16:
                    return ", ".join(out)
        return ", ".join(out)

    rows = []
    for lid, g in df_topics.groupby("label_id", dropna=False):
        rows.append(
            {
                "label_id": str(lid),
                "label": str(g["label"].iloc[0]),
                "n_reviews": int(g["n_reviews"].sum()),
                "avg_rating": wavg(g, "avg_rating"),
                "growth_delta": float(g["growth_delta"].sum()) if "growth_delta" in g.columns else None,
                "priority_score": float(g["priority_score"].sum()) if "priority_score" in g.columns else None,
                "keywords_str": merge_keywords(g),
                "n_clusters": int(g["topic_id"].nunique()) if "topic_id" in g.columns else int(len(g)),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["priority_score", "n_reviews"], ascending=False).reset_index(drop=True)
    return out


st.set_page_config(page_title="Executive — Product Intelligence", layout="wide")
st.title("Executive Overview")
st.caption("What is broken, how big is it, how fast it is changing, and what to fix first.")

summary_path = st.session_state.get("summary_path")
if not summary_path:
    st.error("summary_path missing. Go to Home and set summary.json path.")
    st.stop()

summary = load_summary(summary_path)
meta = _as_dict(summary.get("meta", {}))
dataset = _as_dict(summary.get("dataset", {}))
release_impact = _as_dict(summary.get("release_impact", {}))
coverage = _as_dict(release_impact.get("coverage", {}))

# KPI row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows (core)", str(dataset.get("rows", "-")))
c2.metric("Avg rating", _fmt_float(dataset.get("avg_rating"), 3))
c3.metric("% rating ≤ 2", _fmt_float(dataset.get("pct_rating_le_2"), 2))
c4.metric("Window mode", str(meta.get("windows_mode", "-")))
c5.metric("Coverage OK", "YES" if bool(coverage.get("coverage_ok", True)) else "NO")

with st.expander("Definitions (so a PM understands)", expanded=False):
    st.markdown(
        "- **Theme** = stakeholder-friendly bucket (label)\n"
        "- **Cluster / topic_id** = model cluster (k-means etc.)\n"
        "- **delta** = latest_window_count - prev_window_count\n"
        "- **growth_rate** = delta / prev (prev=0 => inf)\n"
        "- **avg_rating** lower => worse\n"
        "- **avg_negativity** higher => worse (computed by sentiment module)\n"
    )

st.divider()

df_pr = priority_df(summary)
if df_pr.empty:
    st.info("priority_ranking missing/empty in summary.json.")
    st.stop()

df_theme = aggregate_theme(df_pr)

# Row funnel (why 900k->60k)
st.subheader("Row Funnel (why the dataset shrank)")
funnel = _as_dict(meta.get("row_funnel", {}))
if not funnel:
    st.info("row_funnel not found in meta. (Update pipeline.py to write meta['row_funnel'].)")
else:
    funnel_order = [
        ("n_raw_loaded", "raw loaded"),
        ("n_after_schema", "after schema"),
        ("n_after_sampling", "after sampling"),
        ("n_neg_core_pre_clean", "negative core pre-clean"),
        ("n_neg_core_post_clean", "negative core post-clean"),
        ("n_core_final", "core final"),
    ]
    rows = []
    for k, name in funnel_order:
        if k in funnel:
            rows.append({"stage": name, "rows": int(_safe_int(funnel.get(k), 0))})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=220)

st.divider()

st.subheader("Issue Concentration (Theme Pareto)")
topk = st.slider("Top-K themes", min_value=5, max_value=30, value=12, step=1)

left, right = st.columns([1.15, 1])

df_theme_show = df_theme.head(topk).copy()
with left:
    st.caption("Top themes by volume (aggregated).")
    if not df_theme_show.empty:
        st.bar_chart(df_theme_show.set_index("label")["n_reviews"])
    else:
        st.info("Theme table empty.")

with right:
    st.caption("Severity proxy: avg_rating (lower = worse).")
    if not df_theme_show.empty and "avg_rating" in df_theme_show.columns:
        chart = df_theme_show[["label", "avg_rating"]].set_index("label")
        st.bar_chart(chart)

st.divider()

st.subheader("Theme Drilldown (why one theme appears multiple times)")
sel = st.selectbox("Pick a theme", options=df_theme["label"].tolist()[:50] if not df_theme.empty else ["Unlabeled"])
if sel:
    sub = df_pr[df_pr["label"] == sel].sort_values("n_reviews", ascending=False).copy()
    st.caption("Underlying clusters (topic_id). Multiple clusters for the same theme = model split (k too high) or sub-issues.")
    show_cols = ["topic_id", "n_reviews", "avg_rating", "growth_delta", "growth_rate", "priority_score", "keywords_str"]
    show_cols = [c for c in show_cols if c in sub.columns]
    st.dataframe(sub[show_cols].head(20), use_container_width=True, height=360)

st.divider()

st.subheader("Latest Window vs Previous (Change Detection)")
trends = _as_dict(summary.get("trends", {}))
lvp = _as_dict(trends.get("latest_vs_prev", {}))
if not lvp:
    st.info("No latest_vs_prev found in trends.")
else:
    rows = []
    for k, v in lvp.items():
        d = _as_dict(v)
        tid = _safe_int(k, -1)
        rows.append(
            {
                "topic_id": tid,
                "delta": _safe_float(d.get("delta")),
                "growth_rate": _safe_float(d.get("growth_rate")),
            }
        )
    df_lvp = pd.DataFrame(rows)
    df_lvp = df_lvp.merge(df_pr[["topic_id", "label", "n_reviews", "avg_rating"]], on="topic_id", how="left")
    df_lvp = df_lvp.sort_values("delta", ascending=False)
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Largest increases (delta > 0)")
        st.dataframe(df_lvp.head(15), use_container_width=True, height=320)
    with col2:
        st.caption("Largest decreases (delta < 0)")
        st.dataframe(df_lvp.tail(15).sort_values("delta"), use_container_width=True, height=320)

st.divider()

st.subheader("Release Risk Snapshot")
top_versions = _as_list(release_impact.get("top_versions_by_negativity", []))
if not top_versions:
    st.info("No release impact table in summary.")
else:
    df_rel = pd.DataFrame(top_versions)
    show_cols = [c for c in ["version", "n_reviews", "avg_rating", "avg_negativity", "pct_imputed", "top_topic"] if c in df_rel.columns]
    st.dataframe(df_rel[show_cols].head(12), use_container_width=True, height=320)

st.success("Next: use Issues for evidence + Releases for version-level risk + Exports to generate stakeholder artifacts.")
