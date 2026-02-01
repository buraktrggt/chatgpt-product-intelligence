from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


_VERSION_CANDIDATES = [
    "version",
    "app_version",
    "appversion",
    "client_version",
    "build_version",
    "build",
    "build_number",
    "release",
    "release_version",
    "app_ver",
]

# Eğer elinde böyle bir kolon varsa bazen "1.2025.084 (Android)" gibi geliyor:
_VERSION_CLEAN_RE = re.compile(r"([0-9]+(?:\.[0-9]+){1,4})")


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _extract_semver_like(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    m = _VERSION_CLEAN_RE.search(s)
    if not m:
        return ""
    return m.group(1).strip()


def normalize_version_column(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Ensures df has a clean 'version' column.
    Strategy:
      1) Use config['release_impact']['version_col'] if provided and exists
      2) Else search common candidate columns
      3) Clean values into normalized numeric-dot format when possible
      4) Empty/NaN -> 'unknown'
    """
    df = df.copy()

    cfg = config or {}
    ricfg = (cfg.get("release_impact", {}) or {}) if isinstance(cfg, dict) else {}
    preferred = _safe_str(ricfg.get("version_col")).strip()

    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(_VERSION_CANDIDATES)

    chosen: Optional[str] = None
    lower_map = {c.lower(): c for c in df.columns}

    for c in candidates:
        if not c:
            continue
        # allow case-insensitive matching
        if c in df.columns:
            chosen = c
            break
        if c.lower() in lower_map:
            chosen = lower_map[c.lower()]
            break

    if chosen is None:
        # no source column, create unknown
        df["version"] = "unknown"
        return df

    src = df[chosen]

    # Clean into stable format where possible
    cleaned = src.astype(str).map(_extract_semver_like)
    # if no semver-like found, keep original string (but stripped)
    fallback = src.astype(str).str.strip()
    final = cleaned.where(cleaned.str.len() > 0, fallback)

    # normalize empties -> unknown
    final = final.fillna("").astype(str).str.strip()
    final = final.where(final != "", "unknown")
    final = final.where(final.str.lower() != "nan", "unknown")
    final = final.where(final.str.lower() != "none", "unknown")

    df["version"] = final
    return df
