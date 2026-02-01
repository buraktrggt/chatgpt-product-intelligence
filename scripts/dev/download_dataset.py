"""
Dataset download utility for Product Intelligence pipeline.

This script downloads the raw user review dataset from a predefined source
and stores it in the expected location used by the pipeline:

    data/raw/chatgpt_reviews.csv

The dataset itself is intentionally not committed to version control.
This script exists to make data acquisition explicit and reproducible.
"""

from pathlib import Path
import sys
import requests

# -----------------------------
# Configuration
# -----------------------------

DATA_URL = (
    "https://www.kaggle.com/datasets/ashishkumarak/chatgpt-reviews-daily-updated/chatgpt_reviews.csv"
    # Replace with the actual dataset source URL
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "chatgpt_reviews.csv"

CHUNK_SIZE = 1024 * 1024  # 1 MB


# -----------------------------
# Helpers
# -----------------------------

def download_file(url: str, output_path: Path) -> None:
    """Download a file in streaming mode to avoid memory issues."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with output_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        print(f"[INFO] Dataset already exists: {OUTPUT_FILE}")
        print("[INFO] Delete the file if you want to re-download.")
        return

    print("[INFO] Downloading dataset...")
    print(f"[INFO] Source: {DATA_URL}")
    print(f"[INFO] Destination: {OUTPUT_FILE}")

    try:
        download_file(DATA_URL, OUTPUT_FILE)
    except Exception as exc:
        print(f"[ERROR] Download failed: {exc}")
        sys.exit(1)

    print("[SUCCESS] Dataset downloaded successfully.")


if __name__ == "__main__":
    main()
