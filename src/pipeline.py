from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import load_config
from src.data.io import load_raw_data
from src.data.validate import validate_schema
from src.data.sampling import sample_data

from src.preprocessing.clean import clean_texts
from src.data.version_impute import impute_versions_by_timestamp

from src.embeddings.embed import generate_embeddings

from src.topics.cluster import cluster_embeddings
from src.topics.label import label_topics
from src.topics.merge import propose_topic_merges, apply_topic_merges

from src.sentiment.score import compute_negativity_scores
from src.trends.windows import build_time_windows
from src.trends.drift import compute_topic_trends

from src.release_impact.impact import compute_release_impact
from src.summary.builder import build_structured_summary
from src.reporting.render import render_markdown_report

from src.llm.client import maybe_augment_with_llm
from src.utils.json_safe import to_json_safe


def _norm_version_str(x: Any, unknown_token: str = "unknown") -> str:
    try:
        if x is None:
            return unknown_token
        import pandas as pd
        if pd.isna(x):
            return unknown_token
    except Exception:
        if x is None:
            return unknown_token

    s = str(x).strip()
    if s == "":
        return unknown_token
    s_low = s.lower()
    if s_low in {"nan", "none", "<na>", "na", "null"}:
        return unknown_token
    return s


class ProductIntelligencePipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _canonicalize_version(self, df: Any) -> Any:
        cols_cfg = (self.config.get("data", {}) or {}).get("columns", {}) or {}
        raw_version_col = cols_cfg.get("version", "version")

        vi_cfg = (self.config.get("release_impact", {}) or {}).get("version_impute", {}) or {}
        unknown_token = str(vi_cfg.get("unknown_token", "unknown"))

        if "version" in df.columns:
            df["version"] = df["version"].apply(lambda v: _norm_version_str(v, unknown_token))
            return df

        candidates = [raw_version_col, "appVersion", "reviewCreatedVersion", "version", "app_version"]

        src = None
        for c in candidates:
            if c and c in df.columns:
                src = c
                break

        if src is None:
            df["version"] = unknown_token
            return df

        df["version"] = df[src].apply(lambda v: _norm_version_str(v, unknown_token))
        return df

    def _maybe_choose_k_elbow(self, embeddings: Any) -> Optional[Dict[str, Any]]:
        """
        If topics.kmeans_k == "elbow", compute elbow diagnostics + chosen k.
        Returns dict for meta or None.
        """
        topics_cfg = (self.config.get("topics", {}) or {})
        if str(topics_cfg.get("method", "kmeans")).lower() != "kmeans":
            return None

        k_cfg = topics_cfg.get("kmeans_k", 12)
        if not (isinstance(k_cfg, str) and k_cfg.strip().lower() == "elbow"):
            return None

        elbow_cfg = (topics_cfg.get("elbow", {}) or {})
        from src.topics.elbow import choose_kmeans_k_elbow

        res = choose_kmeans_k_elbow(
            embeddings=embeddings,
            config=self.config,
            k_min=int(elbow_cfg.get("k_min", 6)),
            k_max=int(elbow_cfg.get("k_max", 30)),
            step=int(elbow_cfg.get("step", 2)),
            sample_n=int(elbow_cfg.get("sample_n", 30000)),
            seed=int(elbow_cfg.get("seed", 42)),
        )

        return {
            "chosen_k": int(res.chosen_k),
            "method": str(res.method),
            "sample_n": int(res.sample_n),
            "k_values": list(res.k_values),
            "inertia": list(res.inertia),
        }

    def run(self):
        print("=== ChatGPT Product Intelligence Pipeline Started ===")

        output_dir = Path(self.config["project"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        funnel: Dict[str, int] = {}

        # -----------------------
        # Load + validate
        # -----------------------
        print("Loading raw data...")
        df = load_raw_data(self.config)
        funnel["n_raw_loaded"] = int(len(df))

        print("Validating schema...")
        validate_schema(df, self.config)
        funnel["n_after_schema"] = int(len(df))

        # -----------------------
        # Sampling (optional)
        # -----------------------
        print("Sampling data...")
        df = sample_data(df, self.config)
        funnel["n_after_sampling"] = int(len(df))

        # -----------------------
        # Core selection FIRST (performance)
        # -----------------------
        neg_max = int((self.config.get("data", {}) or {}).get("max_rating_for_negative", 3))
        pos_min = int((self.config.get("data", {}) or {}).get("min_rating_for_positive", 4))

        df_neg = df[df["rating"] <= neg_max].copy()
        df_pos = df[df["rating"] >= pos_min].copy()
        funnel["n_neg_core_pre_clean"] = int(len(df_neg))
        funnel["n_pos_pool_pre_clean"] = int(len(df_pos))

        df_neg["track"] = "negative"
        df_pos["track"] = "positive"

        print(f"Negative track rows (core): {len(df_neg):,}")
        print(f"Positive pool rows (metadata only): {len(df_pos):,}")

        df_core = df_neg

        # -----------------------
        # Clean text on core
        # -----------------------
        print("Preprocessing texts (core)...")
        df_core = clean_texts(df_core, self.config)
        funnel["n_neg_core_post_clean"] = int(len(df_core))

        # -----------------------
        # Canonicalize version + snapshot original
        # -----------------------
        df_core = self._canonicalize_version(df_core)

        vi_cfg = (self.config.get("release_impact", {}) or {}).get("version_impute", {}) or {}
        vi_enabled = bool(vi_cfg.get("enabled", True))
        unknown_token = str(vi_cfg.get("unknown_token", "unknown"))

        missing_pct = (df_core["version"] == unknown_token).mean() * 100.0
        print(f"[DEBUG] Version missing after canonicalize: {missing_pct:.2f}%")

        df_core["version_original"] = df_core["version"].apply(lambda v: _norm_version_str(v, unknown_token))

        # -----------------------
        # Version imputation
        # -----------------------
        version_impute_stats: Optional[Any] = None
        if vi_enabled:
            print("Imputing unknown versions by timestamp (guard-railed)...")
            df_core, version_impute_stats = impute_versions_by_timestamp(
                df_core,
                version_col="version",
                time_col=str(vi_cfg.get("time_col", "timestamp")),
                unknown_token=unknown_token,
                min_version_samples=int(vi_cfg.get("min_version_samples", 80)),
                prior_power=float(vi_cfg.get("prior_power", 1.0)),
                min_posterior=float(vi_cfg.get("min_posterior", 0.60)),
                hard_window_days=int(vi_cfg.get("hard_window_days", 14)),
            )
            print(
                f"Version impute: unknown_before={version_impute_stats.n_unknown_before}, "
                f"imputed={version_impute_stats.n_imputed}, "
                f"unknown_after={version_impute_stats.n_unknown_after}, "
                f"imputed_share_of_unknown={version_impute_stats.imputed_share_of_unknown:.2f}%"
            )
        else:
            print("Version impute disabled by config.")
            df_core["_version_imputed"] = False

        funnel["n_core_final"] = int(len(df_core))

        # -----------------------
        # Time windows
        # -----------------------
        print("Building time windows...")
        windows = build_time_windows(df_core, self.config)

        # -----------------------
        # Embeddings
        # -----------------------
        print("Generating embeddings...")
        embeddings = generate_embeddings(df_core, self.config)

        # -----------------------
        # Elbow choose k (if enabled)
        # -----------------------
        elbow_meta = self._maybe_choose_k_elbow(embeddings)
        if elbow_meta:
            print(f"[ELBOW] k chosen = {elbow_meta['chosen_k']} via {elbow_meta['method']}")

        # -----------------------
        # Topic modeling
        # -----------------------
        print("Clustering topics...")
        if elbow_meta:
            topic_ids = cluster_embeddings(embeddings, self.config, method="kmeans", kmeans_k=int(elbow_meta["chosen_k"]))
        else:
            topic_ids = cluster_embeddings(embeddings, self.config)

        # -----------------------
        # Optional topic merge
        # -----------------------
        merge_cfg = (self.config.get("topics", {}) or {}).get("merge", {}) or {}
        merge_enabled = bool(merge_cfg.get("enabled", True))
        merge_threshold = float(merge_cfg.get("threshold", 0.88))

        merge_res = propose_topic_merges(
            embeddings=embeddings,
            topic_ids=topic_ids,
            enabled=merge_enabled,
            threshold=merge_threshold,
        )

        if getattr(merge_res, "enabled", False) and getattr(merge_res, "merges", None):
            print(f"Merging topics (threshold={merge_res.threshold}): {len(merge_res.merges)} groups")
            topic_ids = apply_topic_merges(topic_ids, merge_res.old_to_new)
        else:
            print("Topic merge: no merges applied")

        # -----------------------
        # Labeling
        # -----------------------
        print("Labeling topics...")
        topic_labels = label_topics(df_core, topic_ids, self.config)

        # -----------------------
        # Negativity (single source of truth)
        # -----------------------
        print("Computing negativity scores...")
        negativity_scores = compute_negativity_scores(df_core, self.config)
        df_core["negativity"] = negativity_scores

        # -----------------------
        # Trends + release impact
        # -----------------------
        print("Computing topic trends...")
        trends = compute_topic_trends(df_core, topic_ids, windows, self.config)

        print("Computing release impact...")
        release_impact = compute_release_impact(df_core, topic_ids, self.config)

        # -----------------------
        # Summary
        # -----------------------
        print("Building structured summary...")
        summary = build_structured_summary(
            df=df_core,
            topics=topic_ids,
            topic_labels=topic_labels,
            negativity_scores=negativity_scores,
            trends=trends,
            release_impact=release_impact,
            config=self.config,
            strengths=None,
            embeddings=embeddings,
        )

        # -----------------------
        # Meta hygiene
        # -----------------------
        meta = summary.setdefault("meta", {})
        meta["negative_track_rows_pre_clean"] = int(funnel.get("n_neg_core_pre_clean", 0))
        meta["positive_pool_rows_pre_clean"] = int(funnel.get("n_pos_pool_pre_clean", 0))

        meta["n_input_rows_raw"] = int(funnel.get("n_raw_loaded", 0))
        meta["n_after_schema"] = int(funnel.get("n_after_schema", 0))
        meta["n_after_sampling"] = int(funnel.get("n_after_sampling", 0))
        meta["n_core_rows_post_clean"] = int(funnel.get("n_neg_core_post_clean", 0))
        meta["n_core_rows_final"] = int(funnel.get("n_core_final", 0))
        meta["row_funnel"] = funnel

        meta["topic_merge_enabled"] = bool(merge_enabled)
        meta["topic_merge_threshold"] = float(merge_threshold)
        meta["topic_merge_groups"] = getattr(merge_res, "merges", [])

        if elbow_meta:
            meta["kmeans_elbow"] = elbow_meta

        meta["version_impute_enabled"] = bool(vi_enabled)
        if version_impute_stats is not None:
            meta["version_impute"] = {
                "unknown_before": int(version_impute_stats.n_unknown_before),
                "imputed": int(version_impute_stats.n_imputed),
                "unknown_after": int(version_impute_stats.n_unknown_after),
                "imputed_share_of_unknown_pct": float(version_impute_stats.imputed_share_of_unknown),
                "method": str(getattr(version_impute_stats, "method", "timestamp_posterior")),
            }

        # -----------------------
        # Optional LLM augmentation
        # -----------------------
        print("Optionally augmenting with LLM...")
        summary = maybe_augment_with_llm(summary, self.config)

        # -----------------------
        # Save JSON
        # -----------------------
        json_path = output_dir / "summary.json"
        safe_summary = to_json_safe(summary)
        json_path.write_text(
            json.dumps(safe_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Summary JSON saved to: {json_path}")

        # -----------------------
        # Render report
        # -----------------------
        print("Rendering markdown report...")
        report_path = render_markdown_report(summary, self.config)

        print("=== Pipeline finished successfully ===")
        print(f"Report saved to: {report_path}")

        return summary


def run_pipeline():
    config = load_config("configs/config.yaml")
    pipeline = ProductIntelligencePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
