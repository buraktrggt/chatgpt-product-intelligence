from __future__ import annotations

import re
from typing import Dict, Any

import pandas as pd

try:
    from langdetect import detect
except ImportError:
    detect = None


GENERIC_PRAISE = {
    "good", "nice", "best", "great", "excellent", "awesome", "amazing",
    "super", "perfect", "love", "ok", "okay"
}


def _basic_clean(text: str, *, keep_unicode: bool = True) -> str:
    """
    keep_unicode=True  -> keeps non-latin characters (recommended for global reviews)
    keep_unicode=False -> legacy mode: keep only [a-z0-9] punctuation (aggressive)
    """
    if text is None:
        return ""

    text = str(text).lower()

    # Remove URLs / emails
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    if keep_unicode:
        # Keep letters/numbers from all scripts + basic punctuation
        # (Avoid nuking Turkish/Arabic/Indic etc.)
        text = re.sub(r"[^\w\s\.\,\!\?\-]", " ", text, flags=re.UNICODE)
    else:
        # Aggressive legacy behavior: latin only
        text = re.sub(r"[^a-z0-9\s\.\,\!\?\-]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_generic_praise_short(text_clean: str, max_words: int = 3) -> bool:
    words = re.findall(r"[a-z]+", text_clean)
    if len(words) == 0:
        return True
    if len(words) <= max_words:
        return all(w in GENERIC_PRAISE for w in words)
    return False


def _is_english(text_for_detect: str) -> bool:
    if detect is None:
        return True
    try:
        return detect(text_for_detect) == "en"
    except Exception:
        return False


def clean_texts(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Clean + filter texts.
    Produces 'text_clean' and prints an audit of what was dropped.

    IMPORTANT: For global reviews, keep_unicode=True is strongly recommended.
    """
    df = df.copy()

    preprocessing_cfg = (config.get("preprocessing", {}) or {}) if isinstance(config, dict) else {}
    filtering_cfg = (config.get("filtering", {}) or {}) if isinstance(config, dict) else {}

    keep_unicode = bool(preprocessing_cfg.get("keep_unicode", True))

    before = len(df)

    # 1) basic clean
    df["text_clean"] = df["text"].astype(str).map(lambda t: _basic_clean(t, keep_unicode=keep_unicode))

    # 2) length filter (chars)
    min_len = int(preprocessing_cfg.get("min_text_length", 5))
    mask_len = df["text_clean"].str.len() >= min_len
    dropped_len = int((~mask_len).sum())
    df = df.loc[mask_len].copy()

    # 3) word-count filter
    min_words = int(filtering_cfg.get("min_words", 0))
    dropped_words = 0
    if min_words > 0:
        wc = df["text_clean"].str.split().str.len()
        mask_wc = wc >= min_words
        dropped_words = int((~mask_wc).sum())
        df = df.loc[mask_wc].copy()

    # 4) drop generic praise short
    dropped_praise = 0
    if bool(filtering_cfg.get("drop_praise_short", False)):
        mask = ~df["text_clean"].apply(lambda t: _is_generic_praise_short(t, max_words=3))
        dropped_praise = int((~mask).sum())
        df = df.loc[mask].copy()

    # 5) drop non-english
    dropped_lang = 0
    if bool(filtering_cfg.get("drop_non_english", False)):
        def keep_row(t: str) -> bool:
            if len(t) < 20:
                return True
            return _is_english(t)

        mask = df["text_clean"].apply(keep_row)
        dropped_lang = int((~mask).sum())
        df = df.loc[mask].copy()

    df = df.reset_index(drop=True)
    after = len(df)

    print(
        "Preprocessing audit: "
        f"kept {after:,} / {before:,} "
        f"(dropped_len={dropped_len:,}, dropped_min_words={dropped_words:,}, "
        f"dropped_praise={dropped_praise:,}, dropped_non_english={dropped_lang:,}, "
        f"keep_unicode={keep_unicode}, min_len={min_len}, min_words={min_words})."
    )

    return df
