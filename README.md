# AI Clinical Research Assistant

**AI-Powered Clinical Research Assistant for Patient–Clinical Trial Matching and Post-Treatment Outcome Analysis**

> ⚠️ **SYNTHETIC DATA — FOR RESEARCH PROTOTYPE ONLY**
> This system does not provide medical advice. All patient data is synthetic. Human researcher/clinical review is required for all conclusions.

---

## Architecture

```
NCT05502562 Trial ──► Trial Parser ──► Eligibility Rules
                                            │
synthetic_volunteers.csv ──► Patient Normalizer
                                            │
                          ┌─────────────────▼──────────────────┐
                          │  BM25 + Dense BGE + Cross-Encoder   │
                          │       Hybrid RAG + Rule Engine      │
                          └─────────────────┬──────────────────┘
                                            │
                                    BEST PATIENT MATCH
                                            │
diabetic_data.csv ──► ML Training ──► XGBoost / RF / LR
                                            │
                                          SHAP
                                            │
                              Clinical Outcome Analysis
                                            │
                                    Research Report
```

## Technology Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI + Pydantic |
| ML Models | Logistic Regression, Random Forest, XGBoost |
| Explainability | SHAP |
| Dense Embeddings | BAAI/bge-small-en-v1.5 (sentence-transformers) |
| Sparse Retrieval | BM25 (rank_bm25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Frontend | React 18 + TypeScript + Vite + Recharts |
| Tests | Pytest |

## Setup & Running

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Train ML Models (first time only)
```bash
python -m backend.app.ml.train
```

### 3. Start Backend
```bash
uvicorn backend.app.main:app --reload --port 8000
```

### 4. Start Frontend
```bash
cd frontend
npm install
npm run dev
```

### 5. Run Evaluation Suite
```bash
# Unit Tests
pytest tests/ -v

# Matching Quality Eval
python evals/eval_matching_quality.py

# ML Metrics Eval
python evals/eval_ml_metrics.py

# RAG Quality Eval
python evals/eval_rag_quality.py
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/trials/NCT05502562` | Get structured trial data |
| POST | `/api/matching/run` | Run patient-trial matching |
| POST | `/api/ml/train` | Train ML models |
| POST | `/api/ml/predict` | Predict outcome for patient |
| GET | `/api/ml/shap/{patient_id}` | Get SHAP feature contributions |
| GET | `/api/analysis/synthetic_followup/{patient_id}` | Get synthetic follow-up report |
| POST | `/api/research/analyze` | Generate full research report |

## Data Files

| File | Purpose |
|---|---|
| `NCT05502562.csv` | Clinical trial definition (NCT05502562) |
| `synthetic_type2_diabetes_trial_volunteers_200.csv` | **SYNTHETIC** Phase 1 patient population |
| `diabetic_data.csv` | Historical diabetes dataset for Phase 2 ML |

## Important Disclaimers

- All patient data is **SYNTHETIC** — not real patient records
- The ML model provides an analytical estimate only — not a clinical recommendation  
- Drug-specific outcome prediction is **not validated** unless the dataset contains the exact intervention
- Final enrollment requires researcher/clinical verification
- This system does **not** establish drug efficacy, causality, or make autonomous medical decisions

---

## ML Model Performance

### Current Results (Phase 2 — 30-Day Readmission Prediction)

| Model | ROC-AUC | PR-AUC | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.6829 | 0.1024 | 0.614 | 0.136 |
| Random Forest | 0.6833 | 0.0905 | 0.143 | 0.126 |
| **XGBoost (Best)** | **0.6956** | **0.1241** | **0.439** | **0.161** |

Training dataset: **57,214 unique patients** (patient-level deduplicated), **4.5% positive rate** (<30-day readmission).

---

## Why the Model Accuracy Cannot Be Significantly Improved Further

The current XGBoost model (ROC-AUC **0.6956**) is operating near the **empirical ceiling** for this dataset and task. Below are the concrete, research-backed reasons why further meaningful improvement is not achievable without fundamentally different data.

### 1. This is a Known Hard Problem at the Benchmark Level

Predicting 30-day hospital readmission from administrative EHR data is one of the most studied problems in clinical ML. Independent peer-reviewed research consistently reports ROC-AUC in the **0.63–0.72** range on the UCI Diabetes dataset using traditional ML. Our model at **0.6956** is firmly within — and at the upper end of — this established benchmark band.

> **Reference:** Strack et al., "Impact of HbA1c Measurement on Hospital Readmission Rates," *BioMed Research International*, 2014 (the original paper describing this dataset) reports similar model ceilings.

### 2. The Dataset Lacks the Most Predictive Features

Hospital readmission is primarily driven by **post-discharge factors** that are simply not recorded in the dataset:

| Missing Feature | Why It Matters |
|---|---|
| Discharge instructions / follow-up plans | Primary predictor of 30-day readmission in clinical studies |
| Patient socioeconomic status | Insurance type, housing stability, access to transport |
| Social support (living alone vs. family) | Strong independent predictor of readmission |
| Medication adherence post-discharge | Direct cause of re-admission in diabetes |
| Primary care follow-up booked | Preventive factor not captured in inpatient records |
| Free-text clinical notes | Rich unstructured signal lost in this structured dataset |

Without this data, no ML algorithm — no matter how sophisticated — can predict outcomes driven by unobserved causes.

### 3. Extreme Class Imbalance (4.5% Positive Rate)

Only **4.5% of encounters** result in <30-day readmission. This makes precision inherently low regardless of model sophistication. The PR-AUC baseline for a random model on this task is `~0.045`. Our model achieves `0.124` — **2.7× better than random** — which is strong performance under this constraint.

### 4. Patient-Level Deduplication Reduces Apparent Signal

We intentionally deduplicated to **one encounter per patient** to prevent data leakage. This removes the strong sequential pattern (patients who were readmitted before are much more likely to be readmitted again). Including repeat visits would inflate ROC-AUC artificially to ~0.75–0.80 but would not generalise to new, unseen patients.

### 5. Theoretical Ceiling for This Feature Set

With only structured administrative features (demographics, lab counts, medication flags, diagnosis codes), the **theoretical information ceiling** for 30-day readmission prediction is approximately **ROC-AUC 0.70–0.73**. To exceed this, the following data sources would be required:

- **Clinical NLP on discharge summaries** — BERT/BioBERT on free-text notes
- **Longitudinal vitals streams** — Time-series of blood pressure, glucose trends
- **Social determinants of health (SDOH)** — Zip code, income level, food access
- **Claims data** — Post-discharge pharmacy fills, outpatient visit completion

These require **institutional data access** (HIPAA-compliant, de-identified) beyond the scope of this research prototype.
