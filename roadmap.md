# Restaurant Inspection Early-Warning: 2-Week Roadmap (with PyTorch)

> **Goal (in one line):** Ship a weekly-updated, explainable risk score for restaurants likely to receive violations at their next inspection, with an AWS data/ML pipeline, a small API, and a dashboard — enhanced with a **PyTorch NLP + multi-modal model**.

---

## 0) Story you’ll demo
- **Data → Decisions:** City inspections + Yelp reviews → features → PyTorch model (tabular + review text) → **calibrated risk score** → **dashboard & API**.
- **Ops:** Automated **weekly** pipeline running ETL, training, and batch scoring.
- **UX:** Map/table of top-risk venues; each venue page shows **reason codes** (SHAP/feature importances), historic violations, and review snippets.

---

## 1) System architecture (high level)
**Ingest** → **Lake** → **ETL/Features** → **Modeling (LightGBM baseline + PyTorch fusion)** → **Calibration** → **Batch scoring** → **Serve/UI** → **Ops/Monitoring**

- **Ingest:** Scheduled job to pull inspection datasets + Yelp reviews (batch).
- **Storage:** **S3** with `raw/ → silver/ → gold/` (Parquet; partitioned by `city`/`dt`).
- **ETL:** **PySpark (Glue)** to clean, dedupe, geocode (if needed), link inspections ↔ Yelp businesses.
- **Features:** Tabular (history, geo, cuisine) + text **embeddings** (PyTorch transformer).
- **Models:**
  - Baseline: **LightGBM** on tabular features.
  - **PyTorch multi-modal**: MLP (tabular) + **DistilBERT** (text) → fusion classifier.
- **Calibration:** **Temperature scaling** in PyTorch for well-calibrated probabilities.
- **Scoring:** Weekly **batch** scores to S3/Parquet (`gold/scores_venue_weekly/`).
- **Serve:** Streamlit on **App Runner**; **Lambda + API Gateway** for `/risk` lookup.
- **Ops:** **Step Functions** orchestration, **CloudWatch** metrics, **Great Expectations** DQ.

---

## 2) Tech stack & tooling
- **AWS:** S3, Glue, Athena (optional), Lambda, Step Functions, SageMaker (batch jobs), App Runner, CloudWatch, IAM.
- **Python:** pandas, PySpark, scikit-learn, **lightgbm**, **torch**, **transformers** (HF), **sentence-transformers**, shap, great_expectations, rapidfuzz, geopandas/geopy (local), pyarrow.
- **Dev-XP:** conda, pytest, black/ruff, pre-commit, GitHub Actions, Docker.

---

## 3) Datasets & scope
- **Inspections:** NYC DOHMH or Chicago Food Inspections.
- **Reviews:** Yelp Open Dataset (business + reviews). Subset by city; cap to last 12–24 months.
- **Weather (stretch):** NOAA daily heat index per city/zip.

**Scope v1:** one city + reviews subset; **batch-only** (no streaming).

---

## 4) Data model & tables
**Raw → Silver → Gold** (all Parquet; UTC timestamps; snake_case columns)

- **silver.venues_dim** — venues with IDs, geocodes, cuisine, Yelp link.
- **silver.inspections** — inspection records.
- **silver.reviews** — Yelp reviews.
- **gold.labels_venue_weekly** — binary labels for next inspection violation.
- **gold.features_tabular_venue_weekly** — engineered features.
- **gold.features_text_embed_venue_weekly** — 512-d text embeddings per venue/week.
- **gold.scores_venue_weekly** — risk scores per venue/week with model version.

---

## 5) Features & label rules (leakage-safe)
- Labels: use only data **up to week_start**.
- Features: past violations, stars avg, review counts, cuisine, density, SHAP-explainable.
- Text: DistilBERT embeddings aggregated weekly.

---

## 6) Modeling plan
- **Baseline (Day 4):** LightGBM tabular.
- **Text (Day 5):** DistilBERT frozen embeddings.
- **Fusion (Day 6):** PyTorch MLP combining both.
- **Calibration (Day 7):** Temperature scaling.

Metrics: **PR-AUC, Lift@Top5%, Brier score**. Explainability via SHAP.

---

## 7) Serving & product
- **Batch scoring:** Weekly risk scores in Parquet.
- **API:** `/risk?venue_id=…` → JSON with calibrated risk + reasons.
- **Dashboard:** Streamlit app with overview, venue, and model pages.

---

## 8) MLOps & QA
- Great Expectations for DQ.
- GitHub Actions cron pipeline.
- Logs, model versioning, drift checks (stretch).

---

## 9) Security & cost
- IAM least privilege.
- Partition pruning, only 12–24 months data.
- Batch (not streaming) inference.

---

## 10) Team plan (roles & rotation)
- **Day-by-day (2 weeks)**:
  - **D1:** Data landing (Sanjay lead).  
  - **D2:** Cleaning & linkage (Pranav lead).  
  - **D3:** Tabular features + labels (Sanjay).  
  - **D4:** LightGBM baseline (Pranav).  
  - **D5:** Text embeddings (Pair).  
  - **D6:** Fusion MLP (Sanjay).  
  - **D7:** Calibration + eval (Pranav).  
  - **D8:** Batch scoring CLI (Sanjay).  
  - **D9:** API + Dashboard (Pranav).  
  - **D10:** Polish + Demo (Pair).

Both rotate lead/shadow daily.

---

## 11) Prerequisites & study plan
- **D1–D3:** S3, Parquet, PySpark, Great Expectations, fuzzy matching.
- **D3–D4:** Imbalanced classification, LightGBM basics, SHAP.
- **D5–D7:** PyTorch tensors, nn.Module, HuggingFace basics, calibration.
- **D8–D10:** Streamlit, FastAPI/Lambda, CI/CD, App Runner basics.

---

## 12) Acceptance criteria
- Valid gold tables with schemas.
- Model PR-AUC above baseline; calibrated probs (ECE < 0.05).
- Weekly scores Parquet with versioning.
- Dashboard loads <2s; API returns JSON.
- README + Model Card + Runbook present.

---

## 13) Documentation templates
- **README:** Problem, architecture diagram, datasets, how to run, results, demo, limitations.
- **Model Card:** Intended use, data, performance, fairness notes, version.
- **Runbook:** ETL → train → score steps, inputs/outputs, KPIs, failure modes.

---

## 14) Directory structure
/infra/ # IaC (Terraform/CDK) – optional in v1
/data_scripts/ # ingest, clean_linkage.py, build_labels.py
/featurization/ # features_tabular.py, embeddings_text.py
/models/ # train_lgbm.py, train_fusion_mlp.py, calibrate.py, score.py
/app/ # streamlit (overview.py, venue.py, model.py)
/utils/ # io.py, eval.py, logging.py, hashing.py
/tests/ # unit + DQ tests
.github/workflows/ci.yml
README.md
MODEL_CARD.md
RUNBOOK.md

yaml
Copy code

---

## 15) Stretch goals
- Weather features.
- Neighborhood graph with PyTorch Geometric.
- Fine-tune DistilBERT lightly.
- Evidently for drift monitoring.