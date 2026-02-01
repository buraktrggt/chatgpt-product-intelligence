from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VersionImputeStats:
    n_unknown_before: int
    n_imputed: int
    n_unknown_after: int
    imputed_share_of_unknown: float
    method: str


def _to_ts(x: Any) -> pd.Timestamp:
    ts = pd.to_datetime(x, errors="coerce", utc=True)
    return ts


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def impute_versions_by_timestamp(
    df: pd.DataFrame,
    version_col: str = "version",
    time_col: str = "timestamp",
    unknown_token: str = "unknown",
    min_version_samples: int = 80,
    prior_power: float = 1.0,
    min_posterior: float = 0.60,
    hard_window_days: int = 14,
) -> Tuple[pd.DataFrame, VersionImputeStats]:
    """
    Impute unknown versions using timestamp alignment.

    Core idea:
      - Learn each known version's time support (min/max) and a smoothed density proxy.
      - For an unknown review at time t, consider versions whose (min-max) cover t,
        expanded by +/- hard_window_days.
      - Score candidates by:
          posterior ∝ prior(version)^prior_power * exp(-|t - median_ts| / scale)
        Normalize posteriors and assign if max posterior >= min_posterior.

    Guard rails:
      - Only versions with >= min_version_samples are eligible.
      - If no eligible candidates, keep unknown.
      - Adds df['_version_imputed'] flag.

    Returns: (df_out, stats)
    """
    df_out = df.copy()

    if version_col not in df_out.columns or time_col not in df_out.columns:
        stats = VersionImputeStats(0, 0, 0, 0.0, "timestamp_impute:missing_columns")
        return df_out, stats

    # normalize timestamp
    df_out[time_col] = df_out[time_col].map(_to_ts)
    df_out = df_out.dropna(subset=[time_col]).copy()

    # normalize version to string
    df_out[version_col] = df_out[version_col].fillna(unknown_token).astype(str)

    is_unknown = df_out[version_col].str.lower() == unknown_token.lower()
    n_unknown_before = int(is_unknown.sum())
    if n_unknown_before == 0:
        stats = VersionImputeStats(0, 0, 0, 0.0, "timestamp_impute:no_unknown")
        df_out["_version_imputed"] = False
        return df_out, stats

    known = df_out[~is_unknown].copy()
    if known.empty:
        stats = VersionImputeStats(n_unknown_before, 0, n_unknown_before, 0.0, "timestamp_impute:no_known")
        df_out["_version_imputed"] = False
        return df_out, stats

    # build per-version stats
    per: Dict[str, Dict[str, Any]] = {}
    for v, g in known.groupby(version_col):
        n = int(len(g))
        if n < min_version_samples:
            continue
        ts = g[time_col].astype("int64").to_numpy()
        per[str(v)] = {
            "n": n,
            "min": int(np.min(ts)),
            "max": int(np.max(ts)),
            "median": float(np.median(ts)),
            "mad": float(np.median(np.abs(ts - np.median(ts)))) + 1.0,  # robust scale
        }

    if not per:
        stats = VersionImputeStats(n_unknown_before, 0, n_unknown_before, 0.0, "timestamp_impute:no_eligible_versions")
        df_out["_version_imputed"] = False
        return df_out, stats

    # priors by frequency among eligible known
    total_n = float(sum(v["n"] for v in per.values()))
    priors = {k: (v["n"] / total_n) for k, v in per.items()}

    # window expansion
    win_ns = int(pd.Timedelta(days=int(hard_window_days)).value)

    df_out["_version_imputed"] = False
    imputed = 0

    # iterate unknown rows
    unk_idx = df_out.index[is_unknown].tolist()
    for idx in unk_idx:
        t = int(df_out.at[idx, time_col].value)

        # candidates: within expanded range
        cands = []
        for v, s in per.items():
            if t < (s["min"] - win_ns) or t > (s["max"] + win_ns):
                continue

            # likelihood proxy: exp(-|t-median|/mad)
            dist = abs(float(t) - float(s["median"]))
            scale = float(s["mad"])
            likelihood = float(np.exp(-dist / scale))

            prior = float(priors.get(v, 0.0)) ** float(prior_power)
            score = prior * likelihood

            cands.append((v, score))

        if not cands:
            continue

        scores = np.array([c[1] for c in cands], dtype=float)
        tot = float(scores.sum())
        if tot <= 0:
            continue

        post = scores / tot
        best_i = int(np.argmax(post))
        best_v = cands[best_i][0]
        best_p = float(post[best_i])

        if best_p >= float(min_posterior):
            df_out.at[idx, version_col] = best_v
            df_out.at[idx, "_version_imputed"] = True
            imputed += 1

    n_unknown_after = int((df_out[version_col].str.lower() == unknown_token.lower()).sum())
    share = (float(imputed) / float(max(1, n_unknown_before))) * 100.0

    stats = VersionImputeStats(
        n_unknown_before=n_unknown_before,
        n_imputed=imputed,
        n_unknown_after=n_unknown_after,
        imputed_share_of_unknown=share,
        method="timestamp_impute:prior*exp(-dist/mad)+window",
    )
    return df_out, stats
