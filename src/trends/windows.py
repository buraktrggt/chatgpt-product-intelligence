from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Any, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class TimeWindow:
    start: pd.Timestamp
    end: pd.Timestamp  # inclusive end in our logic
    name: str


def build_time_windows(df: pd.DataFrame, config: Dict[str, Any]) -> List[TimeWindow]:
    mode = config["windows"]["mode"].lower()

    ts = df["timestamp"].dropna()
    if ts.empty:
        raise ValueError("No valid timestamps found; cannot build time windows.")

    min_ts = ts.min()
    max_ts = ts.max()

    if mode == "monthly":
        return _monthly_windows(min_ts, max_ts)

    if mode == "sliding":
        wdays = int(config["windows"]["sliding"]["window_days"])
        sdays = int(config["windows"]["sliding"]["step_days"])
        return _sliding_windows(min_ts, max_ts, window_days=wdays, step_days=sdays)

    raise ValueError(f"Unsupported windows.mode: {mode}")


def _monthly_windows(min_ts: pd.Timestamp, max_ts: pd.Timestamp) -> List[TimeWindow]:
    windows: List[TimeWindow] = []
    cur = pd.Timestamp(year=min_ts.year, month=min_ts.month, day=1, tz=min_ts.tz)

    while cur <= max_ts:
        next_month = (cur + pd.offsets.MonthBegin(1))
        end = next_month - pd.Timedelta(seconds=1)
        name = cur.strftime("%Y-%m")
        windows.append(TimeWindow(start=cur, end=end, name=name))
        cur = next_month

    return windows


def _sliding_windows(min_ts: pd.Timestamp, max_ts: pd.Timestamp, window_days: int, step_days: int) -> List[TimeWindow]:
    windows: List[TimeWindow] = []

    # We align the last window to end at max_ts (latest time)
    end = max_ts
    start = end - pd.Timedelta(days=window_days)

    i = 0
    while start >= min_ts:
        name = f"w{i}_{start.date()}_{end.date()}"
        windows.append(TimeWindow(start=start, end=end, name=name))

        # step backwards
        end = end - pd.Timedelta(days=step_days)
        start = end - pd.Timedelta(days=window_days)
        i += 1

    # Reverse so windows go from old -> new
    windows = list(reversed(windows))

    print(f"Built {len(windows)} windows (mode=sliding, window_days={window_days}, step_days={step_days}).")
    print(f"Latest window: {windows[-1].name} [{windows[-1].start.date()} -> {windows[-1].end.date()}]")

    return windows
