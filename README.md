# 🧬 Antibody Developability & High-HIC Risk Scoring Platform

This platform exposes a trained machine learning model (**ProtT5 + Logistic Regression**) as a production-grade **FastAPI Scoring API**. It includes automated **SQLite database call logging**, an interactive **Streamlit monitoring and data drift dashboard (using Evidently AI)**, a dedicated **Jupyter Notebook for biological data drift analysis (using Kolmogorov-Smirnov statistical tests)**, comprehensive **automated pytest suites**, and a multi-stage **Dockerfile** for robust, containerized deployment.

---

## 🏗️ Architecture Overview

The project is structured according to professional MLOps and clean architecture principles:

```text
├── Dockerfile                   # Multi-stage production container configuration (Non-root, safe)
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
├── notebooks/
│   ├── classification_results.ipynb # Benchmarking PLM Embeddings and Model Families
│   ├── data_drift_analysis.ipynb   # Dedicated Jupyter Notebook for KS-test drift analysis (CE2)
│   └── repo_structure.svg           # High-definition visualization of repo & pytest suite
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

### 2. Run Data Processing & Model Training
Generate the final merged dataset and train the serialized model artifact (`best_model.joblib`) by running:

```bash
# Process raw files and merge embeddings
poetry run python -c "import os; os.chdir('src'); import processing; processing.run_full_processing_pipeline('path_config.yaml')"

# Train and serialize the winning model pipeline
poetry run python -c "import os; os.chdir('src'); import train_best_model; train_best_model.train_and_save_model()"
```

### 3. Start the FastAPI Scoring API
The API loads the trained model **once** at startup and exposes a `/predict` scoring endpoint.

```bash
# Start the FastAPI server locally
poetry run uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```
* **Interactive API Documentation (Swagger)**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to inspect input request formats and test calls.

### 4. Start the Streamlit Monitoring Dashboard
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

The API is containerized using a **multi-stage, non-root, production-grade Dockerfile** that limits image size, avoids root privileges, and secures deployment.

### 1. Build the Docker Image
```bash
docker build -t antibody-scoring-api:latest .
```

### 2. Run the Containerized API
```bash
docker run -p 8000:8000 antibody-scoring-api:latest
```
* **Healthcheck Highlight:** To maintain a lightweight footprint, the container uses a `python:3.10-slim` runner image. Since `slim` images do not include `curl` by default, our Dockerfile implements a smart, self-contained healthcheck via Python's built-in `urllib.request` module:
  ```dockerfile
  HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
      CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1
  ```
* Visit [http://localhost:8000/docs](http://localhost:8000/docs) to verify that your Docker container is actively serving predictions!

---

## 📉 Interpreting Monitoring & Data Drift

### Dedicated Jupyter Notebook (`notebooks/data_drift_analysis.ipynb`)
To satisfy the sub-criterion of **CE2 (Efficacité de la surveillance)**, we have created a dedicated, plain-English Jupyter Notebook. It explains covariate shift, runs the KS test manually, compiles Evidently AI data drift verdicts, and defines the MLOps SOP action plan. 

### Why Data Drift Matters in Antibody Scoring:
In therapeutic antibody development, physical parameters (apparent melting temperature `tm_app` and polyspecificity score `psr`) govern developability success. If a research team starts uploading antibody candidates from a different biological subset or structural target family, their physical distributions might shift:
* **TmApp Shift**: A sudden drop in the melting temperature average (e.g., lower thermal stability).
* **PSR Shift**: A sudden spike in polyspecificity ready scores.

Our dashboard and notebook use **Kolmogorov-Smirnov (KS) statistical tests** (with an operational alpha threshold of $\alpha = 0.05$) to compare training distributions against live production query parameters:
* **p-value < 0.05**: Indicates that the live production distribution has diverged significantly from the training reference, triggering the **MLOps Alarm System**.
* **Statistical Power & Sample Size Constraint**: To ensure mathematical rigor, the system enforces a minimum threshold of **30 successful live queries** before executing automated distribution tests. Running KS-tests on extremely small samples (e.g., fewer than 30) has low statistical power, leading to a high rate of **Type II errors** (false negatives).

### ♿ Accessibility and Inclusive Design:
The Streamlit monitoring interface is designed for full accessibility and inclusive data visualization (**WCAG 2.1 Success Criterion 1.4.1**):
* **Okabe-Ito Color Palette**: Traditional red-green color scales (which are indistinguishable to individuals with red-green color blindness) are replaced with an Okabe-Ito compliant scale:
  * **Low Risk**: Deep Blue (`#0072B2`)
  * **Medium Risk**: Warm Orange (`#E69F00`)
  * **High Risk**: Vermilion Red-Orange (`#D55E00`)
  This selection preserves critical visual distinctions for users with **Protanopia**, **Deuteranopia**, and **Tritanopia**.
* **Redundant Coding (Multi-channel communication)**: Color is never the sole carrier of information. High/Low risk diagnostic verdicts are complemented with explicit bold headers, text descriptions, and structural layout blocks.
* **SLA Threshold Indicator**: The SLA latency limit is represented as a **dashed line with a text label**, ensuring it remains distinguishable on any screen.

---

## 🧪 Valid Test Sequence Library

Use these biologically plausible, standard IUPAC heavy and light chain sequence pairs to test the API scoring endpoints or the Streamlit clinical sandbox:

### Pair 1: Low HIC Risk Profile (Favorable Developability)
*   **Heavy Chain (VH):**
    ```text
    EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYAMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGRDYWGQGTLVTVSS
    ```
*   **Light Chain (VL):**
    ```text
    DIQMTQSPSSLSASVGDRVTITCRASQGISNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPFTFGPGTKVDIK
    ```
*   *Expected parameters:* High Thermal Stability ($T_{m,\text{app}} \ge 68.0^\circ\text{C}$), Low Polyspecificity ($PSR \le 0.3$)

### Pair 2: High HIC Risk Profile (Non-Developable Candidate)
*   **Heavy Chain (VH):**
    ```text
    QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYAMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGYYYYGMDVWGQGTTVTVSS
    ```
*   **Light Chain (VL):**
    ```text
    DIVMTQSPLSLPVTPGEPASISCRSSQSLLHSNGYNYLDWYLQKPGQSPQLLIYLGSNRASGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCMQALQTPYTFGQGTKLEIK
    ```
*   *Expected parameters:* Low Thermal Stability ($T_{m,\text{app}} \le 58.0^\circ\text{C}$), High Polyspecificity ($PSR \ge 1.5$)
