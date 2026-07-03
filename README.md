# 🧬 Antibody Developability & High-HIC Risk Scoring Platform

This platform exposes a trained machine learning model (**ProtT5 + Logistic Regression**) as a production-grade **FastAPI Scoring API**. It includes automated **SQLite database call logging**, an interactive **Streamlit monitoring and data drift dashboard (using Evidently AI)**, comprehensive **automated pytest suites**, and a multi-stage **Dockerfile** for robust containerized deployment.

---

## 🏗️ Architecture Overview

The project is structured according to professional MLOps and clean architecture principles:

```text
├── Dockerfile                   # Multi-stage production container configuration
├── pyproject.toml               # Unified project dependency file (Poetry)
├── poetry.lock                  # Lockfile ensuring environment reproducibility
├── .gitignore                   # Prevents committing caches, OS files, and Parquet data
├── .github/
│   └── workflows/
│       └── ci-cd.yml            # CI/CD pipeline automation (Linter -> Training -> Pytest -> Docker)
├── data/
│   ├── antibodies_raw_dataset.xlsx  # Raw Shehata experimental dataset
│   ├── esm2_embeddings.parquet      # ESM2 Pre-calculated embeddings
│   ├── prott5_embeddings.parquet    # ProtT5 Pre-calculated embeddings
│   ├── antiberty_embeddings.parquet # AntiBERTy Pre-calculated embeddings
│   └── final_dataset.parquet        # Processed merged dataset (generated locally, gitignored)
├── src/
│   ├── app.py                       # FastAPI Scoring Endpoint with Built-in SQLite Logging
│   ├── dashboard.py                 # Streamlit Metrics & Evidently AI Data Drift Dashboard
│   ├── processing.py                # Data Cleaning & Embeddings Merger
│   ├── modeling.py                  # Model Registration & Hyperparameter Optimization
│   ├── train_best_model.py          # Winning model trainer & serializer
│   ├── path_config.yaml             # Relative paths for data processing
│   ├── models_config_cls.yaml       # Hyperparameter space for models
│   └── best_model.joblib            # Serialized trained model pipeline (gitignored)
└── tests/
    ├── test_pipeline.py             # Unit tests for cleaning, engineering, and pipelines
    └── test_app.py                  # API integration, validation guards, and lookup tests
```

---

## 🚀 Quick Start Guide

### 1. Installation and Virtualenv Setup
This project uses **Poetry** to manage dependencies. Ensure you have Python 3.10+ installed.

```bash
# 1. Install all dependencies inside a local virtual environment
poetry install

# 2. Activate the Poetry shell
poetry shell
```

### 2. Run Data Processing & Model Training (Étape 1 & 2)
Generate the final merged dataset and train the serialized model artifact (`best_model.joblib`) by running:

```bash
# Process raw files and merge embeddings
poetry run python -c "import os; os.chdir('src'); import processing; processing.run_full_processing_pipeline('path_config.yaml')"

# Train and serialize the winning model pipeline
poetry run python -c "import os; os.chdir('src'); import train_best_model; train_best_model.train_and_save_model()"
```

### 3. Start the FastAPI Scoring API (Étape 2)
The API loads the trained model **once** at startup and exposes a `/predict` scoring endpoint.

```bash
# Start the FastAPI server locally
poetry run uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```
* **Interactive API Documentation (Swagger)**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to inspect input request formats and test calls.

### 4. Start the Streamlit Monitoring Dashboard (Étape 3)
The dashboard reads query logs directly from the SQLite database (`production_logs.db`) and runs **Evidently AI** to compare live incoming queries against training reference data.

```bash
# Launch the monitoring dashboard
poetry run streamlit run src/dashboard.py
```
* **Interactive View**: Open [http://localhost:8501](http://localhost:8501) to explore live call latency timeseries, predicted probability histograms, and feature-by-feature Kolmogorov-Smirnov statistical drift tests.
* **Auto-Seeding**: If the SQLite database is empty, the dashboard **automatically seeds** 50 realistic historical requests (including a subset representing drifted physical traits) so that your charts and reports are populated immediately.

---

## 🧪 Automated Testing (Pytest)

We have written **11 comprehensive automated tests** under the `tests/` directory to guarantee operational safety.

* **Pipeline Tests (`tests/test_pipeline.py`)**: Checks raw excel parsing, footer slicing, IUPAC amino acid sequence cleaning (gap/dash stripping), HIC target encoding, and model pipeline steps.
* **API Tests (`tests/test_app.py`)**: Tests FastAPI health check, valid exact sequence lookups, fallback fuzzy matching (nearest-neighbor search), missing field validation, invalid data types, out-of-range physical property alerts, and non-amino-acid character rejections.

Run the test suite with verbose reporting:
```bash
poetry run pytest -v
```

---

## 🐳 Containerization (Docker Ready)

The API is containerized using a **multi-stage, non-root, production-grade Dockerfile** that limits image size and maximizes container security.

### 1. Build the Docker Image
```bash
docker build -t antibody-scoring-api:latest .
```

### 2. Run the Containerized API
```bash
docker run -p 8000:8000 antibody-scoring-api:latest
```
* Visit [http://localhost:8000/docs](http://localhost:8000/docs) to verify that your Docker container is actively serving predictions!

---

## 📉 Interpreting Monitoring & Data Drift

### Why Data Drift Matters in Antibody Scoring:
In therapeutic antibody development, physical parameters (apparent melting temperature `tm_app` and polyspecificity score `psr`) govern developability success. If a research team starts uploading antibody candidates from a different biological subset or structural target family, their physical distributions might shift:
* **TmApp Shift**: A sudden drop in the melting temperature average (e.g., lower thermal stability).
* **PSR Shift**: A sudden spike in polyspecificity ready scores.

Our dashboard uses **Evidently AI** to perform **Kolmogorov-Smirnov (KS) statistical tests** (with a threshold of $\alpha = 0.05$) to compare training distributions against live production query parameters:
* **p-value < 0.05**: Indicates that the live production distribution has diverged significantly from the training reference. Evidently flags this column as **"Drifted"**.
* **Overlap Histograms**: The dashboard plots overlapping area charts between reference and live data, allowing the team to visually verify the shift and determine if the model needs to be retrained or calibrated.
