import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(path: str) -> Dict[str, Any]:
    """
    Load YAML config file and return as dict.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    validate_config(config)

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Basic sanity checks for config structure.
    """
    required_top_keys = [
        "project",
        "data",
        "preprocessing",
        "embeddings",
        "topics",
        "windows",
        "scoring",
        "llm",
        "reporting",
    ]

    for key in required_top_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: '{key}'")

    # Check data paths
    raw_path = Path(config["data"]["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data file does not exist: {raw_path}\n"
            f"Please place your dataset at this path or update configs/config.yaml"
        )

    # Check sample size
    sample_size = config["data"].get("sample_size", None)
    if sample_size is not None:
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError("data.sample_size must be a positive integer or null")

    # Check window mode
    window_mode = config["windows"]["mode"]
    if window_mode not in ("monthly", "sliding"):
        raise ValueError("windows.mode must be either 'monthly' or 'sliding'")

    # Check topic method
    topic_method = config["topics"]["method"]
    if topic_method not in ("hdbscan", "kmeans"):
        raise ValueError("topics.method must be either 'hdbscan' or 'kmeans'")

    # Check LLM config
    if not isinstance(config["llm"]["enabled"], bool):
        raise ValueError("llm.enabled must be boolean")

    print("Config loaded and validated successfully.")
