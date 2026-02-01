# pages/02_Issues.py
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
    return "-" if v is None else f"{v:.{nd}f}"


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

    if "keywords" in df.columns:
        df["keywords"] = df["keywords"].apply(lambda x: x if isinstance(x, list) else [])
        df["keywords_str"] = df["keywords"].apply(lambda xs: ", ".join([str(x) for x in xs[:12]]))
    else:
        df["keywords_str"] = ""

    if "label" not in df.columns:
        df["label"] = ""
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    df.loc[df["label"] == "", "label"] = "Unlabeled"

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
                "priority_score": float(g["priority_score"].sum()) if "priority_score" in g.columns else None,
                "keywords_str": merge_keywords(g),
                "n_clusters": int(g["topic_id"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["priority_score", "n_reviews"], ascending=False).reset_index(drop=True)
    return out


def _coerce_examples_schema(ex_any: Any) -> Dict[str, List[Dict[str, Any]]]:
    """
    Supports:
      A) dict schema: {"representative":[...], "most_negative":[...], "recent":[...]}
      B) list schema: [{"kind":"representative", ...}, {"kind":"most_negative", ...}, ...]
    Returns always dict with keys representative/most_negative/recent.
    """
    if isinstance(ex_any, dict):
        out = {}
        for k in ["representative", "most_negative", "recent"]:
            out[k] = _as_list(ex_any.get(k, []))
        return out

    if isinstance(ex_any, list):
        rep, neg, rec = [], [], []
        for item in ex_any:
            it = _as_dict(item)
            kind = str(it.get("kind", "")).strip().lower()
            if kind == "representative":
                rep.append(it)
            elif kind == "most_negative":
                neg.append(it)
            elif kind == "recent":
                rec.append(it)
        return {"representative": rep, "most_negative": neg, "recent": rec}

    return {"representative": [], "most_negative": [], "recent": []}


def _render_example_card(item: Dict[str, Any], idx: int):
    txt = str(item.get("text", "")).strip()
    rating = item.get("rating", None)
    ts = str(item.get("timestamp", "")).strip()
    ver = str(item.get("version", "")).strip()

    header = f"{idx}. rating={rating} | version={ver if ver else '-'}"
    st.markdown(f"**{header}**")
    if ts:
        st.caption(ts)
    st.write(txt[:900] + ("..." if len(txt) > 900 else ""))


st.set_page_config(page_title="Issues — Product Intelligence", layout="wide")
st.title("Issues Drilldown")
st.caption("Themes → clusters → evidence (representative + most negative + recent).")

summary_path = st.session_state.get("summary_path")
if not summary_path:
    st.error("summary_path missing. Go to Home and set summary.json path.")
    st.stop()

summary = load_summary(summary_path)
df_pr = priority_df(summary)
if df_pr.empty:
    st.info("priority_ranking missing/empty.")
    st.stop()

df_theme = aggregate_theme(df_pr)
topic_examples = _as_dict(summary.get("topic_examples", {}))

left, right = st.columns([1.05, 1.25])

with left:
    st.subheader("Theme View (stakeholder)")
    st.caption("Themes are aggregated; n_clusters shows why the same theme appears multiple times at cluster-level.")
    show = df_theme.head(30).copy()
    st.dataframe(
        show[["label", "n_reviews", "n_clusters", "avg_rating", "priority_score", "keywords_str"]],
        use_container_width=True,
        height=560,
    )

with right:
    st.subheader("Theme → Evidence")
    theme = st.selectbox("Pick a theme", options=df_theme["label"].tolist()[:60] if not df_theme.empty else ["Unlabeled"])

    if theme:
        clusters = df_pr[df_pr["label"] == theme].sort_values("n_reviews", ascending=False).head(12).copy()
        st.markdown("#### Underlying clusters (topic_id)")
        show_cols = ["topic_id", "n_reviews", "avg_rating", "growth_delta", "growth_rate", "priority_score", "keywords_str"]
        show_cols = [c for c in show_cols if c in clusters.columns]
        st.dataframe(clusters[show_cols], use_container_width=True, height=300)

        st.markdown("#### Evidence")
        if not topic_examples:
            st.info("topic_examples missing in summary.json. Ensure summary builder writes it.")
        else:
            n_topics = st.slider("How many top clusters to show evidence for", 1, 5, 2)
            per_kind = st.slider("Per kind (representative/negative/recent)", 1, 5, 3)

            top_ids = clusters["topic_id"].tolist()[:n_topics]
            for tid in top_ids:
                key = str(int(tid))
                ex_any = topic_examples.get(key, [])
                ex = _coerce_examples_schema(ex_any)

                st.markdown(f"**Topic {tid} — evidence**")

                colA, colB, colC = st.columns(3)
                with colA:
                    st.caption("Representative")
                    for i, it in enumerate(_as_list(ex["representative"])[:per_kind], 1):
                        _render_example_card(_as_dict(it), i)

                with colB:
                    st.caption("Most negative")
                    for i, it in enumerate(_as_list(ex["most_negative"])[:per_kind], 1):
                        _render_example_card(_as_dict(it), i)

                with colC:
                    st.caption("Recent")
                    for i, it in enumerate(_as_list(ex["recent"])[:per_kind], 1):
                        _render_example_card(_as_dict(it), i)

st.divider()