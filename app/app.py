# app.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
import streamlit as st


DEFAULT_SUMMARY_PATH = Path(
    str(PROJECT_ROOT / "reports" / "exports" / "summary.json")
)
DEFAULT_REPORT_MD_PATH = Path(
    str(PROJECT_ROOT / "reports" / "exports" / "product_health_report.md")
)


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


def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False


def _health_badge(ok: bool) -> str:
    return "✅ OK" if ok else "⚠️ RISK"


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
        df["keywords_str"] = df["keywords"].apply(lambda xs: ", ".join([str(x) for x in xs[:10]]))
    else:
        df["keywords"] = [[] for _ in range(len(df))]
        df["keywords_str"] = ""

    if "label" not in df.columns:
        df["label"] = ""
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].replace("", "Unlabeled")
    return df


def release_df(summary: Dict[str, Any]) -> pd.DataFrame:
    rel = summary.get("release_impact", {})
    rel = rel if isinstance(rel, dict) else {}
    top = rel.get("top_versions_by_negativity", [])
    top = top if isinstance(top, list) else []
    df = pd.DataFrame(top)
    if df.empty:
        return df

    if "n_reviews" in df.columns:
        df["n_reviews"] = df["n_reviews"].apply(lambda x: _safe_int(x, 0))
    for c in ["avg_rating", "avg_negativity", "pct_imputed"]:
        if c in df.columns:
            df[c] = df[c].apply(_safe_float)

    if "version" not in df.columns:
        df["version"] = "unknown"
    df["version"] = df["version"].fillna("unknown").astype(str)
    return df


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="ChatGPT Product Intelligence", layout="wide")
st.title("ChatGPT Product Intelligence")
st.caption("Home • Executive • Issues • Releases • Exports")

# -----------------------------
# Sidebar: single source of truth for paths
# -----------------------------
with st.sidebar:
    st.header("Data Source")

    if "summary_path" not in st.session_state:
        st.session_state["summary_path"] = str(DEFAULT_SUMMARY_PATH)
    if "report_path" not in st.session_state:
        st.session_state["report_path"] = str(DEFAULT_REPORT_MD_PATH)

    st.session_state["summary_path"] = st.text_input(
        "summary.json path", st.session_state["summary_path"]
    )
    st.session_state["report_path"] = st.text_input(
        "product_health_report.md path (optional)", st.session_state["report_path"]
    )

    st.divider()
    st.markdown("**Navigation**")
    st.caption("Use the Streamlit Pages menu (pages/ folder).")

# -----------------------------
# Load summary
# -----------------------------
summary_path = st.session_state["summary_path"]
report_path = st.session_state["report_path"]

try:
    summary = load_summary(summary_path)
except Exception as e:
    st.error(f"Could not load summary.json: {e}")
    st.stop()

meta = _as_dict(summary.get("meta", {}))
dataset = _as_dict(summary.get("dataset", {}))
release_impact = _as_dict(summary.get("release_impact", {}))
coverage = _as_dict(release_impact.get("coverage", {}))

time_min = str(dataset.get("time_min") or "")
time_max = str(dataset.get("time_max") or "")

# -----------------------------
# KPI Row
# -----------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", str(dataset.get("rows", "-")))
c2.metric("Avg rating", str(dataset.get("avg_rating", "-")))
c3.metric("% rating ≤ 2", str(dataset.get("pct_rating_le_2", "-")))
c4.metric("Time min", time_min if time_min else "-")
c5.metric("Time max", time_max if time_max else "-")

st.divider()

# -----------------------------
# Artifacts
# -----------------------------
st.subheader("Artifacts")

p_sum = Path(summary_path)
p_rep = Path(report_path)

a1, a2, a3, a4 = st.columns(4)
a1.metric("summary.json", "FOUND" if _exists(p_sum) else "MISSING")
a2.metric("report.md", "FOUND" if _exists(p_rep) else "MISSING")
a3.metric("Windows mode", str(meta.get("windows_mode", "-")))
a4.metric("LLM enabled", "YES" if bool(meta.get("llm_enabled", False)) else "NO")

with st.expander("Artifact timestamps", expanded=False):
    rows = []
    for name, p in [("summary.json", p_sum), ("product_health_report.md", p_rep)]:
        if _exists(p):
            stat = p.stat()
            rows.append({"artifact": name, "modified": pd.to_datetime(stat.st_mtime, unit="s")})
        else:
            rows.append({"artifact": name, "modified": None})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=140)

st.divider()

# -----------------------------
# Coverage & Imputation Health
# -----------------------------
st.subheader("Telemetry Coverage & Imputation Health")

coverage_ok = bool(coverage.get("coverage_ok", True))
unknown_orig = _safe_float(coverage.get("pct_unknown_version_original")) or 0.0
unknown_cur = _safe_float(coverage.get("pct_unknown_version_current")) or 0.0
rescued = _safe_int(coverage.get("rescued_unknowns"), 0)
rescued_pct = _safe_float(coverage.get("rescued_unknowns_pct_of_original_unknown")) or 0.0
pct_imputed_rows = _safe_float(coverage.get("pct_imputed_rows")) or 0.0

h1, h2, h3, h4, h5 = st.columns(5)
h1.metric("Coverage status", _health_badge(coverage_ok))
h2.metric("Unknown % (original)", f"{unknown_orig:.2f}%")
h3.metric("Unknown % (current)", f"{unknown_cur:.2f}%")
h4.metric("Rescued unknowns", str(rescued))
h5.metric("Imputed rows %", f"{pct_imputed_rows:.2f}%")

warns = []
max_unknown_allowed = _safe_float(coverage.get("max_unknown_pct_allowed"))
min_known_required = _safe_int(coverage.get("min_known_rows_required"), 0)
n_known_current = _safe_int(coverage.get("n_known_version_current"), 0)

if max_unknown_allowed is not None and unknown_cur > max_unknown_allowed:
    warns.append(
        f"Current unknown % ({unknown_cur:.2f}%) exceeds max_unknown_pct_allowed ({max_unknown_allowed:.2f}%)."
    )
if min_known_required and n_known_current < min_known_required:
    warns.append(
        f"Known-version rows ({n_known_current}) below min_known_rows_required ({min_known_required})."
    )
if pct_imputed_rows > 50.0:
    warns.append(f"High imputation rate: {pct_imputed_rows:.2f}% (bias risk).")

if warns:
    st.warning(" | ".join(warns))
else:
    st.success("Coverage and imputation look acceptable under current guardrails.")

st.divider()

# -----------------------------
# Visual quick preview (THIS fixes "yetersiz görüntü")
# -----------------------------
st.subheader("Visual Preview")

df_pr = priority_df(summary)
df_rel = release_df(summary)

left, right = st.columns(2)

with left:
    st.markdown("**Top Issues by Volume (Top 10)**")
    if df_pr.empty:
        st.info("priority_ranking missing/empty.")
    else:
        tmp = df_pr.head(10).sort_values("n_reviews", ascending=True).set_index("label")
        st.bar_chart(tmp["n_reviews"])

with right:
    st.markdown("**Worst Versions by Negativity (Top 10)**")
    if df_rel.empty or "avg_negativity" not in df_rel.columns:
        st.info("release impact table missing/empty.")
    else:
        tmp = df_rel.head(10).copy()
        tmp = tmp.sort_values("avg_negativity", ascending=True).set_index("version")
        st.bar_chart(tmp["avg_negativity"])

st.divider()

# -----------------------------
# Quick table preview
# -----------------------------
st.subheader("Top Issues Preview (table)")

if df_pr.empty:
    st.info("priority_ranking is empty or missing in summary.json.")
else:
    preview = df_pr.head(8).copy()
    cols = ["topic_id", "label", "n_reviews", "avg_rating", "growth_delta", "growth_rate", "priority_score", "keywords_str"]
    cols = [c for c in cols if c in preview.columns]
    st.dataframe(preview[cols], use_container_width=True, height=280)

st.info(
    "Next: use the Pages menu for deep-dive.\n"
    "- **Executive**: KPI + change detection\n"
    "- **Issues**: theme drilldown + examples\n"
    "- **Releases**: version impact + imputation audit\n"
    "- **Exports**: stakeholder downloads"
)
