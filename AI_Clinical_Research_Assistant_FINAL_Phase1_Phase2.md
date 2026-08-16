# AI Clinical Research Assistant — Final Phase 1 + Phase 2 Implementation Specification

## 1. Project Title

**AI-Powered Clinical Research Assistant for Patient–Clinical Trial Matching and Post-Treatment Outcome Analysis**

---

## 2. Core Objective

Build an AI-powered clinical research assistant with two connected phases:

### Phase 1 — Patient–Trial Matching

The researcher provides a clinical trial file, specifically:

```text
NCT05502562
```

The system must extract the trial's:

- Disease/condition
- Intervention/drug
- Inclusion criteria
- Exclusion criteria
- Age requirements
- Sex requirements
- Laboratory requirements
- Medical-history requirements
- Medication restrictions
- Primary outcomes
- Secondary outcomes
- Trial time frames
- Recruitment/status information where available

The system then analyzes:

```text
synthetic_type2_diabetes_trial_volunteers_200.csv
```

and identifies the **best-matching synthetic patient(s)** for the selected trial.

The matching system must combine:

- Structured rule matching
- Eligibility criteria extraction
- Clinical terminology normalization
- Production RAG
- Semantic retrieval
- BM25
- Turovec
- Dense embeddings
- Cross-encoder reranking
- Deterministic eligibility validation
- Match scoring
- Missing-data detection
- Evidence traceability
- Ragobserve (refer pypi or Pranesh-2005/RagObserve)

The result must be:

```text
Potentially Eligible
+
Match Score
+
Passed Criteria
+
Failed Criteria
+
Unknown Criteria
+
Manual Review Items
+
Evidence
```

---

### Phase 2 — AI Clinical Outcome Analysis

After Phase 1 identifies the best patient–trial match, the selected patient's clinical information is analyzed using:

```text
diabetic_data.csv
```

as the historical diabetes/drug-use dataset for ML development, where appropriate.

The Phase 2 engine should analyze:

- Baseline clinical characteristics
- Drug/intervention information available in the dataset
- Laboratory measurements
- Clinical outcomes
- Follow-up information where available
- Patient risk factors
- Treatment-related variables
- Outcome labels
- Trial-defined endpoints
- Safety/adverse-event information where available
- Historical evidence
- ML predictions
- Model explanations

The final output must be an **evidence-grounded clinical research analysis**, not an autonomous medical decision.

---

# 3. Important Data-Integrity Rule

The three datasets have different purposes.

| File | Purpose |
|---|---|
| `NCT05502562` | Researcher-provided clinical trial/protocol data |
| `synthetic_type2_diabetes_trial_volunteers_200.csv` | Synthetic patient population for Phase 1 patient–trial matching |
| `diabetic_data.csv` | Historical diabetes/drug-use data for Phase 2 ML/clinical outcome analysis |

The system must **never falsely claim** that `diabetic_data.csv` contains validated efficacy results for the exact intervention in `NCT05502562` unless the dataset actually contains that intervention and appropriate outcome labels.

If the dataset does not contain drug-specific treatment-response labels, the system must clearly report:

```text
Drug-specific outcome prediction is not validated from the available dataset.
The ML model provides an analytical estimate based on the available historical population.
```

Likewise, the synthetic patient dataset must be labelled:

```text
SYNTHETIC DATA — FOR RESEARCH PROTOTYPE ONLY
```

---

# 4. End-to-End Workflow

```text
                    RESEARCHER
                        |
                        v
                Upload Trial File
                  NCT05502562
                        |
                        v
              Trial Document Parser
                        |
                        v
             Eligibility Extraction
                        |
                        v
                Structured Criteria
                        |
                        v
      synthetic_type2_diabetes_trial_volunteers_200.csv
                        |
                        v
              Patient Normalization
                        |
                        v
                Production RAG
                        |
            +-----------+-----------+
            |           |           |
           BM25       Dense     Reranker
            |           |           |
            +-----------+-----------+
                        |
                        v
              Eligibility Rule Engine
                        |
                        v
                 Match Scoring
                        |
                        v
              BEST PATIENT MATCH
                        |
                        v
                    PHASE 2
                        |
                        v
                  diabetic_data.csv
                        |
                        v
                 ML Training/Data
                   Preparation
                        |
                        v
            Logistic Regression
                    Random Forest
                     XGBoost
                     LGBM or TABFM
                        |
                        v
                 Best ML Model
                        |
                        v
                       SHAP
                        |
                        v
             Outcome Prediction
                        |
                        v
              Clinical Evidence RAG
                        |
                        v
                Research LLM
                        |
                        v
            Final Research Report
```

---

# 5. PHASE 1 — Clinical Trial Understanding

## 5.1 Input

The researcher uploads:

```text
NCT05502562
```

The application should support:

```text
JSON
CSV
TXT
PDF
DOCX
```

depending on the actual file format provided.

---

# 6. Trial Document Processing

Pipeline:

```text
NCT05502562
      |
      v
Document Parser
      |
      v
Text + Tables
      |
      v
Section Detection
      |
      v
Eligibility Section
Outcome Section
Intervention Section
Study Design Section
      |
      v
LLM Structured Extraction
      |
      v
Pydantic Validation
      |
      v
Structured Trial JSON
```

---

# 7. Trial Structured Representation

Convert the trial into:

```json
{
  "trial_id": "NCT05502562",
  "condition": "Type 2 Diabetes",
  "intervention": {
    "name": "REPLACE_WITH_ACTUAL_INTERVENTION_FROM_TRIAL_FILE",
    "type": "drug"
  },
  "eligibility": {
    "inclusion": [],
    "exclusion": []
  },
  "outcomes": {
    "primary": [],
    "secondary": []
  },
  "study_design": {},
  "timeframes": []
}
```

The actual drug must be extracted from the researcher-provided trial file. Never hard-code a drug name if it is not present in the file.

---

# 8. Eligibility Criteria Extraction Model

Use an LLM with structured output.

Recommended architecture:

```text
LLM
+
Pydantic
+
JSON Schema
```

The LLM converts natural language into structured eligibility rules.

Example:

```json
{
  "criterion_id": "INC-001",
  "type": "inclusion",
  "field": "age",
  "operator": "greater_than_or_equal",
  "value": 18,
  "unit": "years",
  "source_text": "...",
  "source_section": "Eligibility Criteria"
}
```

Example laboratory rule:

```json
{
  "criterion_id": "INC-002",
  "type": "inclusion",
  "field": "hba1c",
  "operator": "between",
  "min": 7.0,
  "max": 10.0,
  "unit": "%",
  "source_text": "...",
  "source_section": "Eligibility Criteria"
}
```

---

# 9. Clinical Terminology Normalization

Patient and trial terminology may differ.

Example:

```text
Trial:
Type 2 Diabetes Mellitus

Patient:
T2DM

Normalized:
Type 2 Diabetes Mellitus
```

Recommended approach:

```text
Rule-based normalization
+
Medical terminology dictionary
+
LLM normalization
+
Optional biomedical NLP model
```

Possible biomedical NLP models/tools:

- scispaCy
- SapBERT
- ClinicalBERT/BioClinicalBERT
- UMLS terminology mapping where licensing/access is appropriate

The system must preserve the original term and normalized term.

---

# 10. Phase 1 Patient Dataset

Input file:

```text
synthetic_type2_diabetes_trial_volunteers_200.csv
```

Expected columns may include:

```text
patient_id
age
sex
diabetes_type
diagnosis_confirmed
hba1c
fasting_glucose_mg_dl
weight_kg
bmi
systolic_bp
diastolic_bp
diabetes_duration_years
current_medications
hypertension
cardiovascular_disease
kidney_disease
retinopathy
pregnancy
allergies
creatinine_mg_dl
egfr_ml_min_1_73m2
alt_u_l
ast_u_l
```

The implementation must inspect the actual CSV schema and map available columns instead of assuming every column exists.

---

# 11. Patient Data Preprocessing

Use:

```text
Pandas
+
NumPy
+
Pydantic
```

Steps:

```text
CSV
 |
 v
Schema Detection
 |
 v
Missing Value Analysis
 |
 v
Type Conversion
 |
 v
Unit Normalization
 |
 v
Categorical Normalization
 |
 v
Clinical Terminology Normalization
 |
 v
Validated Patient Records
```

---

# 12. Missing Data Handling

Never automatically treat missing clinical information as eligibility.

Example:

```text
Trial requirement:
eGFR >= threshold

Patient:
eGFR = NULL
```

Output:

```text
UNKNOWN
```

not:

```text
PASS
```

Possible patient status:

```text
COMPLETE
INCOMPLETE
CONFLICTING
REQUIRES_REVIEW
```

---

# 13. Production RAG — Phase 1

Reuse the existing **Production RAG with PageIndex** architecture.

Recommended components:

```text
Document
   |
   v
Page/Section Indexing
   |
   +--> BM25
   |
   +--> Dense Embedding Search
   |
   v
Hybrid Retrieval
   |
   v
Cross-Encoder Reranking
   |
   v
Top Evidence
```

Recommended models:

### Dense Embedding

```text
BAAI/bge-small-en-v1.5
```

For a biomedical-focused version, benchmark a biomedical embedding model as an alternative.

### Sparse Retrieval

```text
BM25
```

### Reranker

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Vector Database

Recommended:

```text
PostgreSQL + pgvector
```

or an existing compatible vector database.

---

# 14. RAG Responsibilities

RAG should answer:

```text
What does the trial criterion say?

Where is the criterion located?

What is the trial-defined endpoint?

What evidence supports this criterion?

What does the trial document state?
```

RAG should NOT independently decide:

```text
Patient is eligible.
```

The deterministic rule engine makes that decision based on the structured criterion and patient value.

---

# 15. Deterministic Eligibility Rule Engine

Implement a rule engine in Python.

Possible outputs:

```text
PASS
FAIL
UNKNOWN
MANUAL_REVIEW
```

Example:

```text
Criterion:
Age >= 18

Patient:
52

Result:
PASS
```

Example:

```text
Criterion:
Pregnancy = exclusion

Patient:
Pregnancy = Yes

Result:
FAIL
```

Example:

```text
Criterion:
eGFR >= X

Patient:
Missing

Result:
UNKNOWN
```

---

# 16. Hard Eligibility Logic

A mandatory exclusion failure should result in:

```text
NOT_ELIGIBLE
```

A mandatory inclusion failure should result in:

```text
NOT_ELIGIBLE
```

Missing required information:

```text
POTENTIALLY_ELIGIBLE_WITH_REVIEW
```

All mandatory criteria pass:

```text
POTENTIALLY_ELIGIBLE
```

The UI must clearly state:

```text
Final enrollment requires researcher/clinical verification.
```

---

# 17. Phase 1 Match Score

After hard filtering, calculate a ranking score.

Example:

```text
Match Score =
0.60 × Eligibility Compatibility
+
0.15 × Clinical Similarity
+
0.10 × Demographic Compatibility
+
0.10 × Semantic Trial Relevance
+
0.05 × Data Completeness
```

The weights must be configurable.

The score means:

```text
How well this synthetic patient matches the selected trial criteria.
```

It does NOT mean:

```text
Probability of treatment success.
```

---

# 18. Clinical Similarity Model

For ranking patients, use a combination of:

```text
Rule-based feature matching
+
Normalized numeric distance
+
Dense embedding similarity
```

Optional model:

```text
Sentence-BERT / biomedical sentence embedding
```

For structured patient records, a hybrid feature approach is preferred over blindly embedding the entire patient CSV row.

---

# 19. Phase 1 Output

Example:

```json
{
  "trial_id": "NCT05502562",
  "best_match": {
    "patient_id": "DM-147",
    "eligibility_status": "POTENTIALLY_ELIGIBLE",
    "match_score": 95.4,
    "passed": 12,
    "failed": 0,
    "unknown": 1,
    "manual_review": 1
  }
}
```

Also return the top 5 or top 10 candidates.

Example:

```text
Rank 1 — DM-147 — 95.4
Rank 2 — DM-082 — 93.8
Rank 3 — DM-191 — 91.7
Rank 4 — DM-033 — 89.4
Rank 5 — DM-112 — 87.9
```

---

# 20. Phase 1 Explainability

For every match, display:

```text
WHY THIS PATIENT MATCHED
```

Example:

```text
Age:
52 → PASS

Type 2 Diabetes:
Yes → PASS

HbA1c:
8.4% → PASS

Kidney function:
eGFR 88 → PASS

Pregnancy:
No → PASS

Required variable:
Unknown → MANUAL REVIEW
```

This is more valuable than displaying only a percentage.

---

# 21. PHASE 2 — Clinical Outcome Analysis

Phase 2 begins after Phase 1 selects the best-matching patient.

Input:

```text
Best matched patient:
DM-147

Trial:
NCT05502562

Intervention:
Extracted from trial file

Historical dataset:
diabetic_data.csv
```

---

# 22. Phase 2 Dataset

The primary historical dataset is:

```text
diabetic_data.csv
```

Use it for:

```text
ML training
Feature analysis
Historical outcome analysis
Medication analysis
Diabetes population modelling
```

The actual schema must be inspected before model training.

Do not assume that this file contains:

```text
the exact intervention
the exact trial population
pre/post drug laboratory measurements
```

unless the data confirms it.

---

# 23. Phase 2 Dataset Validation

Pipeline:

```text
diabetic_data.csv
       |
       v
Schema Inspection
       |
       v
Missingness Analysis
       |
       v
Duplicate Detection
       |
       v
Data-Type Validation
       |
       v
Drug/Medication Identification
       |
       v
Outcome Identification
       |
       v
Feature Engineering
```

Generate a data profile:

```text
Rows
Columns
Missing %
Unique values
Numeric distributions
Medication categories
Outcome distribution
```

---

# 24. Drug/Intervention Detection

Extract the actual intervention from:

```text
NCT05502562
```

Then search `diabetic_data.csv` for corresponding medication/intervention information.

Normalize:

```text
Brand name
Generic name
Abbreviations
Medication categories
```

Use:

```text
Rule-based mapping
+
Drug terminology dictionary
+
LLM only for ambiguous normalization
```

If the exact intervention is absent:

```text
EXACT DRUG NOT PRESENT IN HISTORICAL DATA
```

Do not pretend another medication is equivalent.

---

# 25. ML Target Definition

The preferred target is:

```text
Trial endpoint achievement
```

or another clearly defined clinical outcome that can be reliably derived from the historical dataset.

Example:

```text
target = 1
```

if the endpoint is achieved.

```text
target = 0
```

if the endpoint is not achieved.

If the historical dataset cannot produce a valid endpoint label, define an alternative validated outcome and clearly disclose that it is not the trial endpoint.

---

# 26. ML Feature Engineering

Possible features, depending on actual dataset availability:

```text
Age
Sex
Baseline HbA1c
Baseline glucose
BMI
Weight
Blood pressure
Diabetes duration
Medication history
Kidney function
Creatinine
ALT
AST
Comorbidities
Prior utilization
Other relevant laboratory variables
```

Use only features available before the prediction time point.

Avoid data leakage.

---

# 27. Data Leakage Prevention

The model must not use future information to predict an outcome.

Incorrect:

```text
Predict outcome
using follow-up HbA1c
```

Correct:

```text
Predict outcome
using baseline information
```

Example:

```text
Baseline features
       |
       v
ML prediction
       |
       v
Follow-up
       |
       v
Observed outcome
```

---

# 28. ML Models

Train and compare at least three models.

## Model 1 — Logistic Regression

Purpose:

```text
Interpretable baseline model
```

Use:

```text
scikit-learn LogisticRegression
```

Advantages:

- Simple
- Interpretable
- Strong baseline
- Useful for calibration comparison

---

## Model 2 — Random Forest

Purpose:

```text
Nonlinear tree-based baseline
```

Use:

```text
sklearn.ensemble.RandomForestClassifier
```

Advantages:

- Handles nonlinear relationships
- Robust to feature interactions
- Easy benchmark

---

## Model 3 — XGBoost

Primary model:

```text
XGBClassifier
```

Use XGBoost when the dataset and labels are appropriate.

Reasons:

- Strong tabular performance
- Nonlinear relationships
- Feature interactions
- Good performance on heterogeneous clinical data
- SHAP compatibility

---

# 29. Optional Regression Model

If `diabetic_data.csv` contains an appropriate continuous outcome, train:

```text
XGBRegressor
```

Example:

```text
Predicted change in HbA1c
```

Only use regression when the dataset actually provides valid continuous outcome labels.

---

# 30. Model Training Pipeline

```text
diabetic_data.csv
       |
       v
Clean
       |
       v
Feature Engineering
       |
       v
Train / Validation / Test Split
       |
       v
Preprocessing
       |
       +-------------------+
       |                   |
       v                   v
Logistic Regression   Random Forest
       |                   |
       +---------+---------+
                 |
                 v
              XGBoost
                 |
                 v
       Model Evaluation
                 |
                 v
       Select Best Model
```

Use a patient-level split when multiple records from the same patient exist.

---

# 31. Recommended Preprocessing

For Logistic Regression:

```text
Numerical imputation
+
StandardScaler
+
OneHotEncoder
```

For Random Forest/XGBoost:

```text
Appropriate missing-value handling
+
Categorical encoding
+
Numerical validation
```

Use:

```text
sklearn Pipeline
ColumnTransformer
```

to prevent preprocessing leakage.

---

# 32. Class Imbalance

If the target is imbalanced, evaluate:

```text
Class weights
SMOTE only where scientifically appropriate
Threshold tuning
PR-AUC
```

Do not rely only on accuracy.

---

# 33. ML Evaluation Metrics

For classification:

```text
ROC-AUC
PR-AUC
Accuracy
Precision
Recall
F1
Specificity
Sensitivity
Confusion Matrix
Brier Score
Calibration Curve
```

Most important:

```text
ROC-AUC
PR-AUC
Recall
Precision
F1
Calibration
```

Choose the evaluation metric based on the research use case.

---

# 34. Cross Validation

Use:

```text
Stratified K-Fold
```

for classification when appropriate.

Example:

```text
5-fold cross-validation
```

Use a separate final test set whenever the dataset size permits.

---

# 35. Hyperparameter Optimization

For XGBoost, tune:

```text
n_estimators
max_depth
learning_rate
subsample
colsample_bytree
min_child_weight
reg_alpha
reg_lambda
```

Use:

```text
Optuna
```

or:

```text
RandomizedSearchCV
```

Avoid excessively complex tuning when the dataset is small.

---

# 36. SHAP Explainability

For the selected tree-based model:

```text
XGBoost
+
SHAP
```

Generate:

```text
Global Feature Importance
Local Patient Explanation
SHAP Summary Plot
SHAP Waterfall Plot
```

For example:

```text
PATIENT DM-147

Prediction:
78%

Top contributors:
Baseline HbA1c       +
BMI                   +
Diabetes duration     -
Medication history   +
Age                   +
```

State clearly:

```text
SHAP explains model behavior.
It does not establish clinical causality.
```

---

# 37. Phase 2 Clinical Report Input

The system should additionally support a follow-up report for the selected patient.

Example:

```text
DM-147_followup_report.pdf
```

The report may contain:

```text
Patient ID
Treatment
Follow-up date
HbA1c
Fasting glucose
Weight
BMI
Blood pressure
Creatinine
eGFR
Adverse events
Serious adverse events
Physician observations
```

For the prototype, this can be a synthetic document labelled:

```text
SYNTHETIC CLINICAL REPORT — FOR RESEARCH PROTOTYPE ONLY
```

---

# 38. Clinical Document AI

Pipeline:

```text
PDF/DOCX
   |
   v
PyMuPDF
   |
   v
Text + Tables
   |
   v
LLM Structured Extraction
   |
   v
Pydantic
   |
   v
Clinical JSON
```

Example:

```json
{
  "patient_id": "DM-147",
  "follow_up_week": 40,
  "treatment": "EXTRACTED_FROM_REPORT",
  "measurements": {
    "hba1c": 7.1,
    "fasting_glucose_mg_dl": 135,
    "weight_kg": 77,
    "systolic_bp": 132,
    "diastolic_bp": 84
  },
  "adverse_events": [
    {
      "event": "nausea",
      "severity": "mild",
      "serious": false
    }
  ]
}
```

---

# 39. Baseline vs Follow-Up Engine

Use deterministic Python calculations.

Example:

```text
Baseline HbA1c:
8.4%

Follow-up HbA1c:
7.1%

Change:
-1.3 percentage points
```

Weight:

```text
82 kg → 77 kg

Change:
-5 kg

Percentage:
-6.1%
```

Glucose:

```text
172 → 135 mg/dL

Change:
-37 mg/dL
```

Never rely on the LLM to perform these calculations.

---

# 40. Trial Endpoint Engine

Extract the actual endpoint from:

```text
NCT05502562
```

Represent it structurally.

Example:

```json
{
  "endpoint_id": "EP-001",
  "name": "Actual endpoint from trial",
  "measurement": "Actual measurement",
  "timepoint": "Actual trial timepoint",
  "definition": "Actual endpoint definition",
  "source": "NCT05502562"
}
```

The system must not invent endpoint thresholds.

---

# 41. Endpoint Evaluation

Architecture:

```text
Trial Endpoint
+
Patient Baseline
+
Patient Follow-Up
      |
      v
Deterministic Endpoint Engine
      |
      +--> ACHIEVED
      |
      +--> NOT_ACHIEVED
      |
      +--> UNKNOWN
```

Example:

```text
Baseline:
8.4%

Follow-up:
7.1%

Observed change:
-1.3 percentage points

Trial-defined endpoint:
[Actual endpoint]

Result:
ACHIEVED / NOT_ACHIEVED / UNKNOWN
```

---

# 42. Safety Analysis

Create a separate safety engine.

Extract:

```text
Adverse Event
Severity
Seriousness
Onset
Duration
Outcome
Action Taken
```

Example:

```json
{
  "event": "nausea",
  "severity": "mild",
  "serious": false
}
```

Output:

```text
Adverse events:
1

Serious:
0

Reported:
Mild nausea

Causality:
Not established
```

Never automatically state:

```text
The drug caused nausea.
```

Correct:

```text
Nausea was reported during the follow-up period.
Causal attribution requires clinical assessment.
```

---

# 43. Research Evidence RAG

Phase 2 RAG should index:

```text
NCT05502562
Trial protocol
Trial results
Relevant published studies
Drug evidence
Safety evidence
Clinical literature
Guidelines where appropriate
```

Retrieval:

```text
Research Question
      |
      v
BM25
+
Dense Retrieval
      |
      v
Hybrid Search
      |
      v
Cross-Encoder Reranking
      |
      v
Evidence
      |
      v
Research LLM
```

---

# 44. Research LLM

The LLM receives:

```text
Patient facts
+
Baseline measurements
+
Follow-up measurements
+
Calculated changes
+
Trial endpoint
+
Endpoint result
+
ML prediction
+
SHAP explanation
+
Safety findings
+
Retrieved research evidence
```

The LLM generates:

```text
Research summary
Outcome interpretation
Evidence synthesis
Safety summary
Model explanation
Limitations
Missing information
```

The LLM must not invent evidence.

---

# 45. Agentic Research Assistant

Create controlled tools:

```text
get_trial()
extract_trial_criteria()
get_patient()
validate_patient()
match_patient_to_trial()
get_baseline()
get_followup()
calculate_clinical_changes()
get_trial_endpoints()
evaluate_endpoint()
run_ml_prediction()
get_shap_explanation()
analyze_safety()
search_research_evidence()
generate_report()
```

Workflow:

```text
Researcher:
"Analyze the best matched patient."

Agent:
   |
   +--> Retrieve trial
   |
   +--> Retrieve matched patient
   |
   +--> Validate eligibility
   |
   +--> Retrieve historical ML data
   |
   +--> Run selected model
   |
   +--> Generate SHAP
   |
   +--> Read follow-up report
   |
   +--> Calculate observed changes
   |
   +--> Evaluate trial endpoint
   |
   +--> Analyze safety
   |
   +--> Retrieve evidence
   |
   +--> Generate final report
```

---

# 46. Final Research Report

Generate:

```text
============================================================
AI CLINICAL RESEARCH ANALYSIS
============================================================

TRIAL
------------------------------------------------------------
NCT:
NCT05502562

Intervention:
[Extracted from trial]

PATIENT
------------------------------------------------------------
Patient ID:
DM-147

Source:
Synthetic research prototype

PHASE 1 MATCH
------------------------------------------------------------
Eligibility:
Potentially Eligible

Match Score:
95.4

Passed:
12

Failed:
0

Unknown:
1

Manual Review:
1


OBSERVED CLINICAL CHANGES
------------------------------------------------------------

HbA1c:
8.4% → 7.1%

Change:
-1.3 percentage points

Weight:
82 kg → 77 kg

Change:
-5 kg

Fasting Glucose:
172 → 135 mg/dL

Change:
-37 mg/dL


TRIAL ENDPOINT
------------------------------------------------------------

Endpoint:
[Actual NCT05502562 endpoint]

Observed value:
[Calculated value]

Result:
ACHIEVED / NOT ACHIEVED / UNKNOWN

Source:
NCT05502562


ML ANALYSIS
------------------------------------------------------------

Training Dataset:
diabetic_data.csv

Models Compared:
1. Logistic Regression
2. Random Forest
3. XGBoost

Selected Model:
[Best validated model]

Prediction:
[Prediction]

Probability:
[Probability]

Important:
This is a model output based on the available
historical training population. It is not an
autonomous clinical recommendation.


MODEL EXPLANATION
------------------------------------------------------------

SHAP Top Features:

1. [Feature]
2. [Feature]
3. [Feature]
4. [Feature]
5. [Feature]

SHAP explains model behavior and does not establish
causality.


SAFETY ANALYSIS
------------------------------------------------------------

Adverse Events:
[Events]

Serious Adverse Events:
[Events]

Causality:
Not established


EVIDENCE SYNTHESIS
------------------------------------------------------------

Trial Evidence:
[Evidence]

Research Evidence:
[Evidence]

Patient Evidence:
[Evidence]


MISSING INFORMATION
------------------------------------------------------------

[Missing information]


CONFLICTING INFORMATION
------------------------------------------------------------

[Conflicts]


FINAL RESEARCH INTERPRETATION
------------------------------------------------------------

The system identified the above patient as a
potential match to the researcher-provided trial
based on the available synthetic patient data.

The follow-up clinical record produced the observed
changes shown above.

The trial endpoint was evaluated using the actual
endpoint definition extracted from NCT05502562.

The ML component provides an additional analytical
prediction based on diabetic_data.csv and its
available outcome labels.

The retrieved research evidence provides contextual
support for the analysis.

This system does not independently establish drug
efficacy, treatment causality, or a medical decision.

Human researcher/clinical review is required.
============================================================
```

---

# 47. Dashboard Design

## Dashboard 1 — Trial

```text
NCT05502562

Condition
Intervention
Study Phase
Eligibility
Primary Endpoint
Secondary Endpoints
```

---

## Dashboard 2 — Patient Matching

```text
synthetic_type2_diabetes_trial_volunteers_200.csv

Patient Ranking

#1 DM-147
Match Score: 95.4
Potentially Eligible

#2 DM-082
Match Score: 93.8

#3 DM-191
Match Score: 91.7
```

Clicking a patient displays:

```text
PASS
FAIL
UNKNOWN
MANUAL REVIEW
```

with evidence.

---

## Dashboard 3 — Patient Clinical Timeline

```text
Baseline
   |
   v
Treatment
   |
   v
Week 12
   |
   v
Week 24
   |
   v
Week 40
```

Charts:

```text
HbA1c
Weight
Glucose
Blood Pressure
Kidney Function
Adverse Events
```

---

## Dashboard 4 — ML

```text
Dataset:
diabetic_data.csv

Model Comparison

Logistic Regression
ROC-AUC: ...

Random Forest
ROC-AUC: ...

XGBoost
ROC-AUC: ...

Selected:
XGBoost
```

Then:

```text
Patient Prediction
+
Probability
+
SHAP Explanation
```

---

## Dashboard 5 — Evidence

Display:

```text
Patient Evidence
Trial Evidence
Research Evidence
```

Every generated conclusion should be traceable.

---

# 48. Evidence Traceability

Each important output must show:

```text
Conclusion
   |
   +--> Patient source
   |
   +--> Trial source
   |
   +--> Research source
   |
   +--> Calculation
   |
   +--> ML model if applicable
```

Example:

```text
Endpoint ACHIEVED

Patient evidence:
DM-147_followup_report.pdf

Trial evidence:
NCT05502562

Calculation:
Baseline HbA1c - Follow-up HbA1c

Model:
Not used for endpoint calculation
```

This distinction is extremely important.

---

# 49. Technology Stack

## Frontend

```text
React.js
TypeScript
Recharts / Plotly
```

## Backend

```text
Python
FastAPI
Pydantic
```

## Data

```text
Pandas
NumPy
PostgreSQL
pgvector
```

## Trial

```text
ClinicalTrials.gov Open API
```

## Document AI

```text
PyMuPDF
LLM structured extraction
Pydantic validation
```

## RAG

```text
PageIndex
BM25
BGE embeddings
Cross-encoder reranker
pgvector
```

## NLP

```text
scispaCy
SapBERT
ClinicalBERT/BioClinicalBERT
```

Use biomedical NLP only when it improves terminology normalization or clinical entity recognition.

## Machine Learning

```text
scikit-learn
XGBoost
SHAP
Optuna
```

## Deployment

```text
Docker
Docker Compose
```

## Testing

```text
Pytest
Postman
```

---

# 50. Recommended Backend Structure

```text
clinical-research-assistant/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   │
│   │   ├── api/
│   │   │   ├── trial_routes.py
│   │   │   ├── patient_routes.py
│   │   │   ├── matching_routes.py
│   │   │   ├── analysis_routes.py
│   │   │   └── research_routes.py
│   │   │
│   │   ├── services/
│   │   │   ├── clinicaltrials_service.py
│   │   │   ├── trial_parser.py
│   │   │   ├── eligibility_service.py
│   │   │   ├── patient_matching_service.py
│   │   │   ├── document_service.py
│   │   │   ├── endpoint_service.py
│   │   │   ├── safety_service.py
│   │   │   ├── ml_service.py
│   │   │   ├── shap_service.py
│   │   │   ├── rag_service.py
│   │   │   └── research_agent.py
│   │   │
│   │   ├── models/
│   │   │   ├── patient.py
│   │   │   ├── trial.py
│   │   │   ├── eligibility.py
│   │   │   ├── outcome.py
│   │   │   └── report.py
│   │   │
│   │   ├── ml/
│   │   │   ├── preprocess.py
│   │   │   ├── train.py
│   │   │   ├── evaluate.py
│   │   │   ├── predict.py
│   │   │   └── explain.py
│   │   │
│   │   ├── rag/
│   │   │   ├── ingest.py
│   │   │   ├── embeddings.py
│   │   │   ├── bm25.py
│   │   │   ├── hybrid.py
│   │   │   └── reranker.py
│   │   │
│   │   └── utils/
│   │
│   └── tests/
│
├── data/
│   ├── NCT05502562
│   ├── synthetic_type2_diabetes_trial_volunteers_200.csv
│   └── diabetic_data.csv
│
├── models/
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── model_metadata.json
│
├── rag_index/
│
├── frontend/
│
├── docker-compose.yml
└── README.md
```

---

# 51. API Design

## Upload Trial

```http
POST /api/trials/upload
```

Input:

```text
NCT05502562
```

Output:

```json
{
  "trial_id": "NCT05502562",
  "status": "processed"
}
```

---

## Match Patients

```http
POST /api/matching/run
```

Input:

```json
{
  "trial_id": "NCT05502562",
  "patient_dataset": "synthetic_type2_diabetes_trial_volunteers_200.csv"
}
```

Output:

```json
{
  "best_patient_id": "DM-147",
  "match_score": 95.4,
  "status": "POTENTIALLY_ELIGIBLE"
}
```

---

## Train ML

```http
POST /api/ml/train
```

Input:

```json
{
  "dataset": "diabetic_data.csv",
  "target": "endpoint_outcome"
}
```

Output:

```json
{
  "selected_model": "XGBoost",
  "roc_auc": 0.84,
  "f1": 0.79
}
```

These values are examples only; the actual system must calculate them.

---

## Predict

```http
POST /api/ml/predict
```

Input:

```json
{
  "patient_id": "DM-147"
}
```

Output:

```json
{
  "model": "XGBoost",
  "prediction": 1,
  "probability": 0.78
}
```

---

## Upload Follow-Up Report

```http
POST /api/analysis/report
```

Input:

```text
DM-147_followup_report.pdf
```

Output:

```json
{
  "patient_id": "DM-147",
  "status": "processed"
}
```

---

## Final Research Analysis

```http
POST /api/research/analyze
```

Input:

```json
{
  "trial_id": "NCT05502562",
  "patient_id": "DM-147"
}
```

Output:

```json
{
  "status": "completed",
  "report_id": "REPORT-001"
}
```

---

# 52. Final Winning Architecture

```text
                    NCT05502562
                         |
                         v
               Trial Understanding
                         |
                  +------+------+
                  |             |
                  v             v
             LLM Extractor     RAG
                  |             |
                  +------+------+
                         |
                         v
               Structured Criteria
                         |
                         v
 synthetic_type2_diabetes_trial_volunteers_200.csv
                         |
                         v
                 Patient Matching
                         |
           +-------------+-------------+
           |             |             |
        Rules         RAG         Similarity
           |             |             |
           +-------------+-------------+
                         |
                         v
               Eligibility Engine
                         |
                         v
                  Match Ranking
                         |
                         v
                  BEST PATIENT
                         |
                         v
                 PHASE 2 ANALYSIS
                         |
                         v
                 diabetic_data.csv
                         |
              +----------+----------+
              |          |          |
              v          v          v
          Logistic    Random      XGBoost
         Regression   Forest
              |          |          |
              +----------+----------+
                         |
                         v
                  Model Selection
                         |
                         v
                       SHAP
                         |
                         v
                 Outcome Prediction
                         |
                         v
              Follow-up Clinical Report
                         |
                         v
              Document Intelligence
                         |
                         v
             Baseline vs Follow-up
                         |
                         v
                Endpoint Engine
                         |
                         v
                Safety Analysis
                         |
                         v
                  Research RAG
                         |
                         v
                  Research LLM
                         |
                         v
              FINAL RESEARCH REPORT
```

---

# 53. Model Responsibility Matrix

| Component | Model/Technology | Responsibility |
|---|---|---|
| Trial extraction | LLM | Convert trial text to structured criteria |
| Clinical terminology | scispaCy/SapBERT/LLM | Normalize medical terms |
| Dense retrieval | BGE embedding | Semantic retrieval |
| Sparse retrieval | BM25 | Keyword retrieval |
| Reranking | Cross-encoder | Rank retrieved evidence |
| Eligibility | Python rule engine | Deterministic criterion evaluation |
| Match ranking | Hybrid scoring | Rank candidate patients |
| Document extraction | LLM + Pydantic | Extract clinical facts |
| Baseline/follow-up | Python | Calculate changes |
| Endpoint | Python rule engine | Evaluate actual trial endpoint |
| ML baseline | Logistic Regression | Interpretable benchmark |
| ML model | Random Forest | Nonlinear benchmark |
| ML primary | XGBoost | Tabular outcome prediction |
| Explainability | SHAP | Explain ML behavior |
| Evidence retrieval | Production RAG | Retrieve research evidence |
| Research synthesis | LLM | Generate grounded report |
| Final decision | Human researcher | Review and interpret |

---

# 54. Critical Safety and Research Rules

The system must never:

```text
Prescribe a drug
```

```text
Tell a patient to start treatment
```

```text
Claim a drug cured the patient
```

```text
Claim efficacy from one synthetic patient
```

```text
Claim statistical significance from one patient
```

```text
Claim causality from temporal association alone
```

```text
Invent missing laboratory values
```

```text
Invent trial endpoints
```

```text
Invent clinical research evidence
```

```text
Call a patient definitely eligible without human verification
```

The system should instead use:

```text
Potentially Eligible
Observed Improvement
Endpoint Achieved
Endpoint Not Achieved
Unknown
Requires Manual Review
Model Prediction
Evidence Suggests
Causality Not Established
```

---

# 55. Final Project Pitch

Use this description when presenting:

> **Our system is an AI-powered Clinical Research Assistant with two phases. In Phase 1, a researcher provides a clinical trial such as NCT05502562, and our system uses LLM-based eligibility extraction, Production RAG, biomedical terminology normalization, deterministic clinical rules, and hybrid patient matching to identify the best-matching patient from `synthetic_type2_diabetes_trial_volunteers_200.csv`. In Phase 2, the selected patient is analyzed using historical diabetes/drug-use data from `diabetic_data.csv`, an explainable ML pipeline comparing Logistic Regression, Random Forest and XGBoost, SHAP-based model interpretation, clinical-document intelligence, deterministic endpoint analysis, safety analysis, and evidence-grounded RAG. The system then produces an auditable research report showing patient–trial compatibility, observed clinical changes, trial endpoint status, ML prediction, model explanation, safety findings, supporting evidence, and missing information for human researcher review.**

---

# 56. The Most Important Differentiator

Do not position the system as:

```text
Patient → LLM → Trial
```

Position it as:

```text
Trial Understanding
       ↓
Evidence Retrieval
       ↓
Structured Eligibility
       ↓
Deterministic Matching
       ↓
Best Patient
       ↓
Historical ML Analysis
       ↓
Clinical Report Intelligence
       ↓
Endpoint Evaluation
       ↓
Safety Analysis
       ↓
Evidence Retrieval
       ↓
Explainable Research Assistant
       ↓
Auditable Research Report
```

The strongest architectural principle is:

```text
LLM
→ understands documents

RAG
→ retrieves evidence

Rule Engine
→ evaluates deterministic clinical criteria

Python
→ performs calculations

XGBoost
→ predicts from historical structured data

SHAP
→ explains the ML prediction

Research LLM
→ synthesizes the evidence

Human Researcher
→ makes the final research judgment
```

This separation makes the system more reliable, explainable, auditable, and suitable for a clinical-research hackathon prototype.
