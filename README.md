
# Product Intelligence & Health Monitoring System
### Large-Scale User Feedback Analysis (NLP • Time-Aware Analytics)
=======
<img src="assets/demo.gif" width="900" alt="Demo preview" />



\# ChatGPT Product Intelligence

Automated pipeline that turns large-scale app reviews into \*\*prioritized product \& engineering issues\*\*.



\*\*Delivers\*\*

\- Semantic clustering of complaint themes (embeddings + unsupervised learning)

\- Trend detection with sliding windows (weekly/monthly configurable)

\- Release/version impact signals (when metadata exists)

\- Executive-ready outputs: report + JSON summary



\*\*Quickstart\*\*

```bash

pip install -r requirements.txt
python scripts/dev/download_dataset.py
python run_pipeline.py
streamlit run app/app.py
```


python scripts/dev/download\_dataset.py

python run\_pipeline.py



Outputs



reports/exports/product\_health\_report.md



reports/exports/summary.json



Dataset is not included in the repo (size/licensing). Download via the script above.

