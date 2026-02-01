from __future__ import annotations

from typing import Any, Dict, List, Tuple
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# Stopwords
# ============================================================

_BASE_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "while",
    "to", "of", "in", "on", "at", "by", "for", "from", "with", "as", "without",

    "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "mine", "we", "us", "our", "you", "your", "yours",
    "he", "she", "they", "them", "their", "theirs",

    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing",
    "can", "could", "should", "would", "will", "may", "might", "must",

    "not", "no",
    "don", "dont", "doesn", "didn", "isn", "aren", "wasn", "weren",
    "won", "cant", "couldn", "shouldn", "wouldn",

    "all", "some", "any", "many", "much", "more", "most", "less", "least",
    "very", "really", "just", "also", "even", "so", "too",
    "one", "two", "first", "second",

    "have", "has", "had", "having",
    "get", "got", "getting",
    "make", "made", "making",
    "use", "used", "using",

    "good", "great", "best", "bad", "worst", "terrible", "amazing", "awesome",
    "love", "nice", "perfect", "like", "liked", "liking", "better",

    "please", "thanks", "thank",

    # complaint glue
    "because", "want", "why", "what", "only", "always", "sometimes",
    "need", "help", "give", "let",
    "there", "here", "over", "about", "anything", "everything",
}

_DOMAIN_STOP = {
    "app", "apps", "application",
    "chatgpt", "chat", "gpt", "openai",
    "issue", "issues", "problem", "problems",
    "work", "works", "working",
    "time", "times",
}

_STOPWORDS = _BASE_STOP | _DOMAIN_STOP


# ============================================================
# Normalization
# ============================================================

def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"\S+@\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# TF-IDF
# ============================================================

def _build_vectorizer(config: Dict[str, Any]) -> TfidfVectorizer:
    cfg = (config.get("topics", {}) or {}).get("labeling", {}) or {}
    ngram_range = tuple(cfg.get("ngram_range", (1, 3)))
    min_df = int(cfg.get("min_df", 5))
    max_df = float(cfg.get("max_df", 0.80))
    max_features = int(cfg.get("max_features", 40000))
    return TfidfVectorizer(
        stop_words=sorted(_STOPWORDS),
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )


def _is_junk_term(term: str) -> bool:
    t = (term or "").strip()
    if not t or len(t) <= 2:
        return True
    parts = t.split()
    if not parts:
        return True
    if all(p in _STOPWORDS for p in parts):
        return True
    if all(p.isdigit() for p in parts):
        return True
    if len(parts) == 1 and parts[0] in {"thing", "things", "stuff", "today", "now"}:
        return True
    return False


def _top_terms(X, features: np.ndarray, idx: np.ndarray, topk: int = 30) -> List[str]:
    if idx.size == 0:
        return []
    v = X[idx].mean(axis=0)
    v = v.A1 if hasattr(v, "A1") else np.asarray(v).ravel()

    order = np.argsort(v)[::-1]
    out: List[str] = []
    for i in order:
        if v[i] <= 0:
            break
        t = str(features[i]).strip()
        if _is_junk_term(t):
            continue
        out.append(t)
        if len(out) >= topk:
            break
    return out


# ============================================================
# Phrase rules (gated to avoid over-triggering)
# ============================================================

# helpers for gating
_UPDATE_TOKEN = r"(update|updated|new version|latest version|after update|after the update)"
_REGRESSION_TOKEN = r"(broke|broken|worse|regress|regression|stopped|stop working|doesn t work|won t work|crash|crashes|freeze|stuck|error|bug)"

_PHRASE_RULES: List[Tuple[str, str, str]] = [
    (r"\blog\s?in\b|\blogin\b|\bsign\s?in\b", "Login / Sign-in Failure", "login_signin"),
    (r"\bsign\s?up\b|\bcreate account\b|\bregister\b", "Sign-up / Account Creation", "signup_account"),
    (r"\bsubscription\b|\bsubscribe\b|\bpaywall\b|\bpaid\b|\bpay\b|\bmoney\b|\bplus\b", "Subscription / Paywall", "subscription_paywall"),
    (r"\bvoice\b|\baudio\b|\bmic\b|\bmicrophone\b", "Voice Mode / Audio Issues", "voice_audio"),
    (r"\bimage\b|\bimages\b|\bphoto\b|\bpicture\b|\bupload\b|\bsend\b|\bgenerate\b", "Image / Photo Features", "image_features"),
    (r"\bslow\b|\blag\b|\bloading\b|\btakes\b|\blong\b|\bwait\b", "Performance / Slowness", "performance_slow"),

    # gated update regression (must contain update signal AND regression signal)
    (rf"\b{_UPDATE_TOKEN}\b.*\b{_REGRESSION_TOKEN}\b|\b{_REGRESSION_TOKEN}\b.*\b{_UPDATE_TOKEN}\b",
     "Update Regression", "update_regression"),

    (r"\bwrong answer\b|\bincorrect\b|\bhallucinat|\bmisinformation\b", "Incorrect Answers / Reliability", "wrong_answers"),
    (r"\bcrash\b|\bcrashes\b|\bfreeze\b|\bstuck\b|\bhang\b", "Crashes / Freezes", "crash_freeze"),
    (r"\berror\b|\berrors\b|\bbug\b|\bfail\b|\bfailed\b", "Errors / Failures", "errors_failures"),
]

# If we hit generic buckets, try to override with more specific hints
_PRIORITY_HINTS: List[Tuple[str, str, str]] = [
    (r"\blog\s?in\b|\blogin\b|\bsign\s?in\b", "Login / Sign-in Failure", "login_signin"),
    (r"\bsubscription\b|\bpaywall\b|\bplus\b|\bpaid\b", "Subscription / Paywall", "subscription_paywall"),
    (r"\bvoice\b|\baudio\b|\bmic\b", "Voice Mode / Audio Issues", "voice_audio"),
    (r"\bimage\b|\bphoto\b|\bpicture\b|\bupload\b", "Image / Photo Features", "image_features"),
    (r"\bslow\b|\blag\b|\bloading\b", "Performance / Slowness", "performance_slow"),
]


def _slug(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:40] if len(s) > 40 else s


def _label_from_keywords(keywords: List[str]) -> Tuple[str, str]:
    """
    Returns (label, label_id)
    label_id is canonical for stable aggregation.
    """
    if not keywords:
        return "General Product Issue", "general"

    joined = " ".join(keywords[:25])

    # 1) phrase rules
    for pat, label, lid in _PHRASE_RULES:
        if re.search(pat, joined):
            if lid in {"errors_failures", "crash_freeze"}:
                for hpat, hlabel, hlid in _PRIORITY_HINTS:
                    if re.search(hpat, joined):
                        return hlabel, hlid
            return label, lid

    # 2) prefer multi-word phrases
    phrases = [k for k in keywords if (" " in k) and (not _is_junk_term(k))]
    if phrases:
        label = " / ".join(phrases[:2]).title()
        return label, _slug(label)

    # 3) strong unigrams
    strong_uni = [
        k for k in keywords
        if (" " not in k)
        and (k not in _STOPWORDS)
        and (len(k) >= 3)
        and (not k.isdigit())
    ]
    if len(strong_uni) >= 2:
        label = f"{strong_uni[0]} / {strong_uni[1]}".title()
        return label, _slug(label)
    if len(strong_uni) == 1:
        label = strong_uni[0].title()
        return label, _slug(label)

    return "General Product Issue", "general"


# ============================================================
# Public API
# ============================================================

def label_topics(
    df: pd.DataFrame,
    topic_ids: np.ndarray,
    config: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    text_col = "text_clean" if "text_clean" in df.columns else "text"
    texts = df[text_col].astype(str).map(_normalize_text).tolist()

    vec = _build_vectorizer(config)
    X = vec.fit_transform(texts)
    features = np.asarray(vec.get_feature_names_out())

    topic_ids = np.asarray(topic_ids).astype(int).ravel()
    topk = int(((config.get("topics", {}) or {}).get("labeling", {}) or {}).get("topk_terms", 30))

    out: Dict[int, Dict[str, Any]] = {}
    for tid in sorted(set(topic_ids.tolist())):
        idx = np.where(topic_ids == tid)[0]
        kws = _top_terms(X, features, idx, topk=topk)
        label, label_id = _label_from_keywords(kws)
        out[int(tid)] = {"label": label, "label_id": label_id, "keywords": kws}

    return out


def rename_topics_with_llm(
    topic_labels: Dict[int, Dict[str, Any]],
    config: Dict[str, Any],
    **_: Any,
) -> Dict[int, Dict[str, Any]]:
    return topic_labels
