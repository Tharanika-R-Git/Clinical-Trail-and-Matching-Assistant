"""
research_routes.py — Evidence-Grounded Research Report Generation

Generates a complete research report by aggregating:

- Patient-trial match results
- Patient demographics and clinical profile
- Baseline-to-follow-up clinical changes
- Endpoint evaluation
- Safety/adverse-event analysis
- ML prediction
- SHAP explanations
- RAG evidence

No LLM generation is performed here.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from backend.app.services.endpoint_service import EndpointService
from backend.app.services.safety_service import SafetyService
from backend.app.services.ml_service import MLService
from backend.app.services.shap_service import SHAPService
from backend.app.services.rag_service import rag_service
from backend.app.services.trial_parser import TrialParser


router = APIRouter()


# ============================================================
# PROJECT PATHS
# ============================================================

# research_routes.py
#   backend/
#       app/
#           api/
#               research_routes.py
#
# parents[3] = project root

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PATIENT_CSV = PROJECT_ROOT / "synthetic_type2_diabetes_trial_volunteers_200.csv"


# ============================================================
# SERVICES
# ============================================================

_endpoint_service = EndpointService()
_safety_service = SafetyService()
_ml_service = MLService()
_shap_service = SHAPService(_ml_service)
_trial_parser = TrialParser()


# ============================================================
# REQUEST MODEL
# ============================================================

class ReportRequest(BaseModel):
    patient_id: str
    trial_id: str = "NCT05502562"

    match_score: Optional[float] = None

    eligibility_status: Optional[str] = None

    followup_overrides: Optional[Dict[str, Any]] = None

    include_rag_evidence: bool = True

    rag_top_k: int = 5


# ============================================================
# NORMALIZE PATIENT ID
# ============================================================

def normalize_patient_id(value: Any) -> str:
    """
    Makes the following equivalent:

        DM001
        DM-001
        DM_001
        dm001
        dm-001
        DM 001
    """

    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


# ============================================================
# LOAD PATIENT DATASET
# ============================================================

def _load_patient(patient_id: str) -> Dict[str, Any]:

    # --------------------------------------------------------
    # Check CSV
    # --------------------------------------------------------

    if not PATIENT_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Patient dataset not found",
                "expected_path": str(PATIENT_CSV),
                "project_root": str(PROJECT_ROOT),
            },
        )

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    try:
        df = pd.read_csv(PATIENT_CSV)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read patient dataset: {exc}",
        )

    # --------------------------------------------------------
    # Check Patient_ID column
    # --------------------------------------------------------

    if "Patient_ID" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Patient_ID column not found in patient CSV",
                "available_columns": df.columns.tolist(),
            },
        )

    # --------------------------------------------------------
    # Normalize IDs
    # --------------------------------------------------------

    requested_id = normalize_patient_id(patient_id)

    normalized_ids = df["Patient_ID"].apply(normalize_patient_id)

    # --------------------------------------------------------
    # Find patient
    # --------------------------------------------------------

    rows = df[normalized_ids == requested_id]

    # --------------------------------------------------------
    # Patient not found
    # --------------------------------------------------------

    if rows.empty:

        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Patient '{patient_id}' not found",
                "requested_normalized_id": requested_id,
                "available_sample_ids": (
                    df["Patient_ID"]
                    .astype(str)
                    .head(20)
                    .tolist()
                ),
            },
        )

    # --------------------------------------------------------
    # Convert row to dictionary
    # --------------------------------------------------------

    row = rows.iloc[0].to_dict()

    # --------------------------------------------------------
    # Remove NaN values
    # --------------------------------------------------------

    patient = {
        key: (
            None
            if pd.isna(value)
            else value
        )
        for key, value in row.items()
    }

    return patient


# ============================================================
# BUILD FOLLOW-UP DATA
# ============================================================

def _build_followup(
    patient: Dict[str, Any],
    overrides: Optional[Dict[str, Any]]
) -> Dict[str, Any]:

    followup: Dict[str, Any] = {}

    field_map = {

        "HbA1c_percent":
            "HbA1c_Followup_percent",

        "Weight_kg":
            "Weight_Followup_kg",

        "Fasting_Glucose_mg_dL":
            "Fasting_Glucose_Followup_mg_dL",

        "BMI":
            "BMI_Followup",

        "BP_Systolic_mmHg":
            "BP_Systolic_Followup_mmHg",
    }

    # --------------------------------------------------------
    # Use actual follow-up values if present
    # --------------------------------------------------------

    for base_field, followup_field in field_map.items():

        if (
            followup_field in patient
            and patient[followup_field] is not None
        ):

            followup[base_field] = patient[followup_field]

        # ----------------------------------------------------
        # Otherwise simulate follow-up
        # ----------------------------------------------------

        elif (
            base_field in patient
            and patient[base_field] is not None
        ):

            try:

                value = float(patient[base_field])

                if base_field == "HbA1c_percent":
                    new_value = value - 1.2

                elif base_field == "Weight_kg":
                    new_value = value - 4.0

                elif base_field == "Fasting_Glucose_mg_dL":
                    new_value = value - 25.0

                elif base_field == "BMI":
                    new_value = value - 1.4

                elif base_field == "BP_Systolic_mmHg":
                    new_value = value - 3.0

                else:
                    new_value = value

                followup[base_field] = round(
                    new_value,
                    2
                )

            except (ValueError, TypeError):
                pass

    # --------------------------------------------------------
    # Adverse events
    # --------------------------------------------------------

    if "adverse_events" in patient:
        followup["adverse_events"] = patient["adverse_events"]

    if "Adverse_Events" in patient:
        followup["Adverse_Events"] = patient["Adverse_Events"]

    # --------------------------------------------------------
    # Caller overrides
    # --------------------------------------------------------

    if overrides:
        followup.update(overrides)

    return followup


# ============================================================
# FORMAT ENDPOINT SUMMARY
# ============================================================

def _format_endpoint_summary(
    endpoints: Dict[str, Any]
) -> str:

    lines: List[str] = []

    primary = endpoints.get(
        "primary",
        {}
    )

    if primary:

        name = primary.get(
            "endpoint_name",
            "Primary endpoint"
        )

        result = primary.get(
            "result",
            "UNKNOWN"
        )

        change = primary.get(
            "observed_change"
        )

        unit = primary.get(
            "unit",
            ""
        )

        if change is not None:

            try:
                change_str = (
                    f" "
                    f"(observed change: "
                    f"{float(change):+.2f} "
                    f"{unit})"
                )
            except (ValueError, TypeError):

                change_str = (
                    f" "
                    f"(observed change: "
                    f"{change} "
                    f"{unit})"
                )

        else:

            change_str = ""

        lines.append(
            f"  PRIMARY: {name} — "
            f"{result}{change_str}"
        )

    secondary = endpoints.get(
        "secondary",
        []
    )

    for sec in secondary:

        name = sec.get(
            "endpoint_name",
            "Secondary"
        )

        result = sec.get(
            "result",
            "UNKNOWN"
        )

        change = sec.get(
            "observed_change"
        )

        unit = sec.get(
            "unit",
            ""
        )

        if change is not None:

            try:
                change_str = (
                    f" "
                    f"({float(change):+.2f} "
                    f"{unit})"
                )
            except (ValueError, TypeError):

                change_str = (
                    f" "
                    f"({change} {unit})"
                )

        else:

            change_str = ""

        lines.append(
            f"  SECONDARY: {name} — "
            f"{result}{change_str}"
        )

    if not lines:
        return "  No endpoint data available."

    return "\n".join(lines)


# ============================================================
# FORMAT SHAP SUMMARY
# ============================================================

def _format_shap_summary(
    contributions: List[Dict[str, Any]]
) -> str:

    if not contributions:
        return "  No SHAP contributions available."

    lines = []

    for item in contributions[:5]:

        feature = item.get(
            "feature",
            "unknown"
        )

        contribution = item.get(
            "contribution",
            0.0
        )

        try:
            value = float(contribution)

            if value > 0:
                direction = "↑ risk"
            elif value < 0:
                direction = "↓ risk"
            else:
                direction = "neutral"

            lines.append(
                f"  {feature}: "
                f"{value:+.4f} "
                f"({direction})"
            )

        except (ValueError, TypeError):

            lines.append(
                f"  {feature}: "
                f"{contribution}"
            )

    return "\n".join(lines)


# ============================================================
# GENERATE RESEARCH REPORT
# ============================================================

@router.post(
    "/generate",
    response_model=Dict[str, Any]
)
def generate_report(req: ReportRequest):

    # ========================================================
    # 1. LOAD TRIAL
    # ========================================================

    try:

        trial = _trial_parser.parse(
            req.trial_id
        )

    except Exception as exc:

        raise HTTPException(
            status_code=404,
            detail={
                "message": "Unable to load trial",
                "trial_id": req.trial_id,
                "error": str(exc),
            },
        )

    # ========================================================
    # 2. LOAD PATIENT
    # ========================================================

    patient = _load_patient(
        req.patient_id
    )

    # ========================================================
    # 3. BUILD FOLLOW-UP
    # ========================================================

    followup = _build_followup(
        patient,
        req.followup_overrides
    )

    # ========================================================
    # 4. CLINICAL CHANGES
    # ========================================================

    try:

        changes = (
            _endpoint_service
            .calculate_clinical_changes(
                patient,
                followup
            )
        )

    except Exception as exc:

        changes = {
            "error": str(exc)
        }

    # ========================================================
    # 5. ENDPOINT EVALUATION
    # ========================================================

    try:

        endpoints = (
            _endpoint_service
            .evaluate_endpoint(
                patient,
                followup
            )
        )

    except Exception as exc:

        endpoints = {
            "error": str(exc),
            "primary": {},
            "secondary": [],
        }

    # ========================================================
    # 6. SAFETY ANALYSIS
    # ========================================================

    try:

        safety = (
            _safety_service
            .analyze_adverse_events(
                followup
            )
        )

    except Exception as exc:

        safety = {
            "error": str(exc),
            "total_count": 0,
            "serious_count": 0,
            "summary": "Safety analysis unavailable.",
        }

    # ========================================================
    # 7. ML + SHAP
    # ========================================================

    ml_error = None

    try:

        ml_pred = (
            _ml_service
            .predict_patient(patient)
        )

        shap_contribs = (
            _shap_service
            .explain_patient(patient)
        )

    except Exception as exc:

        ml_error = str(exc)

        ml_pred = {
            "model_name": "N/A",
            "prediction": None,
            "probability": None,
        }

        shap_contribs = []

    # ========================================================
    # 8. RAG EVIDENCE
    # ========================================================

    rag_evidence: List[Dict[str, Any]] = []

    if req.include_rag_evidence:

        query = (
            f"{req.trial_id} "
            f"eligibility criteria "
            f"HbA1c {patient.get('HbA1c_percent', '')} "
            f"eGFR {patient.get('eGFR_mL_min_1_73m2', '')} "
            f"semaglutide "
            f"Type 2 Diabetes "
            f"clinical outcomes"
        )

        try:

            rag_evidence = (
                rag_service
                .hybrid_search(
                    query,
                    top_k=req.rag_top_k
                )
            )

        except Exception as exc:

            rag_evidence = [
                {
                    "text": (
                        "RAG retrieval failed: "
                        f"{exc}"
                    ),
                    "score": 0.0,
                    "source": "error",
                }
            ]

    # ========================================================
    # 9. EXTRACT VALUES
    # ========================================================

    hba1c_data = changes.get(
        "hba1c_change",
        {}
    )

    weight_data = changes.get(
        "weight_change",
        {}
    )

    hba1c_change = hba1c_data.get(
        "change"
    )

    weight_change = weight_data.get(
        "change"
    )

    primary_result = (
        endpoints
        .get("primary", {})
        .get("result", "UNKNOWN")
    )

    ml_probability = ml_pred.get(
        "probability"
    )

    # --------------------------------------------------------
    # Safe probability formatting
    # --------------------------------------------------------

    if ml_probability is not None:

        try:

            ml_probability_str = (
                f"{float(ml_probability):.1%}"
            )

        except (ValueError, TypeError):

            ml_probability_str = str(
                ml_probability
            )

    else:

        ml_probability_str = "N/A"

    # ========================================================
    # 10. SAFE CHANGE FORMAT
    # ========================================================

    if hba1c_change is not None:

        try:
            hba1c_change_str = (
                f"{float(hba1c_change):+.2f}%"
            )
        except (ValueError, TypeError):
            hba1c_change_str = str(
                hba1c_change
            )

    else:

        hba1c_change_str = "N/A"

    if weight_change is not None:

        try:
            weight_change_str = (
                f"{float(weight_change):+.2f} kg"
            )
        except (ValueError, TypeError):
            weight_change_str = str(
                weight_change
            )

    else:

        weight_change_str = "N/A"

    # ========================================================
    # 11. ML PREDICTION LABEL
    # ========================================================

    prediction = ml_pred.get(
        "prediction"
    )

    if prediction == 1:

        prediction_label = (
            "Readmission Risk"
        )

    elif prediction == 0:

        prediction_label = (
            "Low Readmission Risk"
        )

    else:

        prediction_label = "N/A"

    # ========================================================
    # 12. EVIDENCE TEXT
    # ========================================================

    evidence_lines = []

    for index, evidence in enumerate(
        rag_evidence[:5],
        start=1
    ):

        text = evidence.get(
            "text",
            ""
        )

        source = evidence.get(
            "source",
            "unknown"
        )

        evidence_lines.append(
            f"  [{index}] "
            f"{str(text)[:200]} "
            f"(source: {source})"
        )

    evidence_text = (
        "\n".join(evidence_lines)
        if evidence_lines
        else "  No evidence retrieved."
    )

    # ========================================================
    # 13. GENERATE NARRATIVE REPORT
    # ========================================================

    generated_at = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    report_text = f"""
CLINICAL RESEARCH REPORT — {req.trial_id}
Generated: {generated_at}
============================================================

PATIENT SUMMARY

  Patient ID : {req.patient_id}
  Age        : {patient.get('Age', 'N/A')} years
  Gender     : {patient.get('Gender', 'N/A')}
  Diabetes   : {patient.get('Diabetes_Type', 'Type 2 Diabetes')}
  HbA1c      : {patient.get('HbA1c_percent', 'N/A')}%
  eGFR       : {patient.get('eGFR_mL_min_1_73m2', 'N/A')} mL/min/1.73m2
  Medication : {patient.get('Current_Medication', 'N/A')}


TRIAL MATCHING

  Trial      : {req.trial_id}
  Title      : {trial.get('title', 'N/A')}
  Status     : {req.eligibility_status or 'N/A'}
  Match Score: {req.match_score if req.match_score is not None else 'N/A'}


CLINICAL CHANGES

  HbA1c Change : {hba1c_change_str}
  Weight Change: {weight_change_str}


ENDPOINT EVALUATION

{_format_endpoint_summary(endpoints)}


SAFETY SUMMARY

  Total AEs   : {safety.get('total_count', 0)}
  Serious AEs : {safety.get('serious_count', 0)}
  Summary     : {safety.get('summary', 'No adverse events recorded.')}


ML OUTCOME PREDICTION

  Model       : {ml_pred.get('model_name', 'N/A')}
  Prediction  : {prediction_label}
  Probability : {ml_probability_str}
  Primary Endpoint: {primary_result}


TOP SHAP FEATURE CONTRIBUTIONS

{_format_shap_summary(shap_contribs)}


EVIDENCE CITATIONS ({len(rag_evidence)} retrieved)

{evidence_text}


DISCLAIMERS

  - Clinical changes are calculated from baseline and follow-up values.
  - Follow-up values may be simulated where actual measurements are unavailable.
  - ML predictions are population-derived estimates.
  - SHAP values explain model behavior and do not establish clinical causality.
  - RAG evidence is provided for research support and requires researcher verification.
  - This report does not constitute medical advice or a regulatory assessment.
""".strip()

    # ========================================================
    # 14. RETURN COMPLETE JSON
    # ========================================================

    return {

        "status": "success",

        "patient_id": req.patient_id,

        "normalized_patient_id":
            normalize_patient_id(
                req.patient_id
            ),

        "trial_id":
            req.trial_id,

        "report_text":
            report_text,

        "report_sections": {

            # ------------------------------------------------
            # Patient
            # ------------------------------------------------

            "patient_profile": {

                "patient_id":
                    req.patient_id,

                "age":
                    patient.get("Age"),

                "gender":
                    patient.get("Gender"),

                "diabetes_type":
                    patient.get(
                        "Diabetes_Type"
                    ),

                "hba1c":
                    patient.get(
                        "HbA1c_percent"
                    ),

                "egfr":
                    patient.get(
                        "eGFR_mL_min_1_73m2"
                    ),

                "medication":
                    patient.get(
                        "Current_Medication"
                    ),
            },

            # ------------------------------------------------
            # Trial matching
            # ------------------------------------------------

            "trial_matching": {

                "trial_id":
                    req.trial_id,

                "trial_title":
                    trial.get(
                        "title"
                    ),

                "eligibility_status":
                    req.eligibility_status,

                "match_score":
                    req.match_score,
            },

            # ------------------------------------------------
            # Clinical changes
            # ------------------------------------------------

            "clinical_changes":
                changes,

            # ------------------------------------------------
            # Endpoints
            # ------------------------------------------------

            "endpoint_evaluation":
                endpoints,

            # ------------------------------------------------
            # Safety
            # ------------------------------------------------

            "safety_summary":
                safety,

            # ------------------------------------------------
            # ML
            # ------------------------------------------------

            "ml_prediction": {

                **ml_pred,

                "model_metrics":
                    (
                        _ml_service
                        .metadata
                        .get(
                            "metrics",
                            {}
                        )
                        if _ml_service.metadata
                        else {}
                    ),
            },

            # ------------------------------------------------
            # SHAP
            # ------------------------------------------------

            "shap_contributions":
                shap_contribs,

            # ------------------------------------------------
            # RAG
            # ------------------------------------------------

            "rag_evidence":
                rag_evidence,
        },

        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "source":
            "research_routes.py — deterministic research report synthesis",

        "ml_error":
            ml_error,
    }


# ============================================================
# TRIAL SUMMARY
# ============================================================

@router.get(
    "/trial-summary/{trial_id}",
    response_model=Dict[str, Any]
)
def get_trial_summary(
    trial_id: str
):

    try:

        trial = _trial_parser.parse(
            trial_id
        )

        return {
            "status": "success",
            "trial_id": trial_id,
            "trial": trial,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=404,
            detail={
                "message":
                    f"Unable to load trial '{trial_id}'",
                "error":
                    str(exc),
            },
        )