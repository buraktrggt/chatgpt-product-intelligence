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


def _fmt_pct(x: Any, nd: int = 2) -> str:
    v = _safe_float(x)
    return "-" if v is None else f"{v:.{nd}f}%"


@st.cache_data(show_spinner=False)
def load_summary(path_str: str) -> Dict[str, Any]:
    p = Path(path_str)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("summary.json must be a JSON object")
    return data


st.set_page_config(page_title="Releases — Product Intelligence", layout="wide")
st.title("Releases & Version Impact")
st.caption("Worst versions + imputation audit + unknown visibility.")

summary_path = st.session_state.get("summary_path")
if not summary_path:
    st.error("summary_path missing. Go to Home.")
    st.stop()

summary = load_summary(summary_path)
release_impact = _as_dict(summary.get("release_impact", {}))

coverage = _as_dict(release_impact.get("coverage", {}))
unk_orig = _as_dict(release_impact.get("unknown_original_bucket", {}))
unk_cur = _as_dict(release_impact.get("unknown_current_bucket", {}))

top_versions = _as_list(release_impact.get("top_versions_by_negativity", []))
df_rel = pd.DataFrame(top_versions)

# Coverage cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Coverage OK", "YES" if bool(coverage.get("coverage_ok", True)) else "NO")
c2.metric("Unknown % (original)", _fmt_pct(coverage.get("pct_unknown_version_original")))
c3.metric("Unknown % (current)", _fmt_pct(coverage.get("pct_unknown_version_current")))
c4.metric("Rescued unknowns", str(_safe_int(coverage.get("rescued_unknowns"), 0)))
c5.metric("Imputed rows %", _fmt_pct(coverage.get("pct_imputed_rows")))

st.divider()

st.subheader("Worst Versions by Negativity")
if df_rel.empty:
    st.info("No release impact table in summary.json.")
    st.stop()

# Types
if "n_reviews" in df_rel.columns:
    df_rel["n_reviews"] = df_rel["n_reviews"].apply(lambda x: _safe_int(x, 0))
for c in ["avg_rating", "avg_negativity", "pct_imputed"]:
    if c in df_rel.columns:
        df_rel[c] = df_rel[c].apply(_safe_float)

show_unknown = st.checkbox("Show version='unknown' audit row (if present)", value=True)

df_show = df_rel.copy()
if not show_unknown and "version" in df_show.columns:
    df_show = df_show[df_show["version"].astype(str).str.lower() != "unknown"].copy()

topn = st.slider("Show top N rows", min_value=5, max_value=30, value=min(15, len(df_show)), step=1)
df_show = df_show.head(topn).copy()

left, right = st.columns([1.35, 1])

with left:
    cols = [c for c in ["kind", "version", "n_reviews", "avg_rating", "avg_negativity", "pct_imputed", "top_topic"] if c in df_show.columns]
    st.dataframe(df_show[cols], use_container_width=True, height=380)

with right:
    st.caption("avg_negativity by version (higher = worse).")
    chart_df = df_show.copy()
    if "version" in chart_df.columns and "avg_negativity" in chart_df.columns:
        # For chart: drop audit unknown row to keep scale clean (optional)
        chart_df = chart_df[chart_df["version"].astype(str).str.lower() != "unknown"].copy()
        if not chart_df.empty:
            st.bar_chart(chart_df.set_index("version")[["avg_negativity"]])
        else:
            st.info("No non-unknown rows to chart.")

st.divider()

st.subheader("Unknown Buckets (Audit)")
b1, b2, b3, b4 = st.columns(4)
b1.metric("Unknown bucket (original) n", str(_safe_int(unk_orig.get("n_reviews"), 0)))
b2.metric("Unknown bucket (original) avg rating", str(unk_orig.get("avg_rating", "-")))
b3.metric("Unknown bucket (current) n", str(_safe_int(unk_cur.get("n_reviews"), 0)))
b4.metric("Unknown bucket (current) avg rating", str(unk_cur.get("avg_rating", "-")))

st.caption(
    "Audit meaning: unknown telemetry may be biased. Compare original vs current unknown buckets. "
    "Imputation should reduce current unknown, but low-confidence rows remain unknown by design."
)

st.divider()