# 🚀 Product Intelligence & Health Monitoring System
### Large-Scale User Feedback Analysis (NLP • Time-Aware Analytics)

---

## 🧭 Project Overview

This project implements a **Product Intelligence system** that analyzes large-scale user feedback to detect product issues, track their evolution over time, and identify regressions.

The system focuses on **problem discovery and degradation monitoring** rather than generic sentiment summarization.

---

## 🧠 Modeling Approach

### Topic Modeling

Topic discovery is performed using **semantic embeddings and clustering**.
The pipeline supports **local LLM-based or embedding-based representations**, depending on configuration.

- LLM usage is **local / self-hosted** and optional
- No dependency on hosted third-party inference services is required
- All modeling steps are reproducible from raw data

---

## 🏗️ Architecture

- Data ingestion & validation (`src/data/`)
- Text preprocessing (`src/preprocessing/`)
- Sentiment scoring (`src/sentiment/`)
- Embeddings & topic discovery (`src/embeddings/`, `src/topics/`)
- Temporal trend analysis (`src/trends/`)
- Release impact analysis (`src/release_impact/`)
- Summary & reporting (`src/summary/`, `src/reporting/`)

---

## 📁 Repository Structure

```
chatgpt-product-intelligence/
├── app/
├── configs/
├── data/
├── reports/
├── scripts/
├── src/
└── run_pipeline.py
```

---

## ▶️ Running

```bash
pip install -r requirements.txt
python scripts/dev/download_dataset.py
python run_pipeline.py
streamlit run app/app.py
```

---

## 📌 Notes

- Absolute, user-specific paths are intentionally avoided
- All file access is project-relative for portability
- Dataset files are excluded from version control

