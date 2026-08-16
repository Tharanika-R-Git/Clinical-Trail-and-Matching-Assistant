from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from pathlib import Path

import pandas as pd
from backend.app.api.patient_routes import _load_patient

from backend.app.services.endpoint_service import EndpointService

from backend.app.services.safety_service import SafetyService
from backend.app.services.ml_service import MLService
from backend.app.services.shap_service import SHAPService


router = APIRouter()


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Project root:
# PEC-Hackathon-Tharanika-main/
#
# This file:
# backend/app/api/analysis_routes.py
#
# parents[0] = api
# parents[1] = app
# parents[2] = backend
# parents[3] = project root

BASE_DIR = Path(__file__).resolve().parents[3]

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PATIENT_CSV = PROJECT_ROOT / "synthetic_type2_diabetes_trial_volunteers_200.csv"


# ============================================================
# SERVICES
# ============================================================

endpoint_service = EndpointService()
safety_service = SafetyService()
ml_service = MLService()
shap_service = SHAPService(ml_service)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class AnalysisRequest(BaseModel):
    patient_id: str
    followup_overrides: Optional[Dict[str, Any]] = None


# ============================================================
# LOAD PATIENT
# ============================================================

def _load_patient(patient_id: str) -> Dict[str, Any]:
    if not PATIENT_CSV.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Patient CSV not found: {PATIENT_CSV}"
        )

    df = pd.read_csv(PATIENT_CSV)

    if "Patient_ID" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Patient_ID column not found",
                "columns": df.columns.tolist()
            }
        )

    def normalize_id(value):
        return (
            str(value)
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
        )

    requested_id = normalize_id(patient_id)

    df["_normalized_id"] = (
        df["Patient_ID"]
        .astype(str)
        .apply(normalize_id)
    )

    print("Requested ID:", patient_id)
    print("Normalized ID:", requested_id)
    print("CSV IDs:", df["Patient_ID"].head(10).tolist())

    rows = df[df["_normalized_id"] == requested_id]

    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Patient '{patient_id}' not found",
                "normalized_requested_id": requested_id,
                "available_ids": df["Patient_ID"].head(20).tolist()
            }
        )

    patient = rows.iloc[0].to_dict()

    patient.pop("_normalized_id", None)

    return {
        key: None if pd.isna(value) else value
        for key, value in patient.items()
    }

# ============================================================
# BUILD FOLLOW-UP DATA
# ============================================================

def _build_followup(
    patient: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    followup = {}

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
    # Use actual follow-up values if available
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

            value = patient[base_field]

            try:

                value = float(value)

                if base_field == "HbA1c_percent":

                    followup[base_field] = round(
                        value - 1.2,
                        2
                    )

                elif base_field == "Weight_kg":

                    followup[base_field] = round(
                        value - 4.0,
                        2
                    )

                elif base_field == "Fasting_Glucose_mg_dL":

                    followup[base_field] = round(
                        value - 25.0,
                        2
                    )

                elif base_field == "BMI":

                    followup[base_field] = round(
                        value - 1.4,
                        2
                    )

                elif base_field == "BP_Systolic_mmHg":

                    followup[base_field] = round(
                        value - 3.0,
                        2
                    )

            except (ValueError, TypeError):

                pass

    # --------------------------------------------------------
    # Caller-provided overrides
    # --------------------------------------------------------

    if overrides:

        followup.update(overrides)

    # --------------------------------------------------------
    # Adverse events
    # --------------------------------------------------------

    if "adverse_events" in patient:

        followup["adverse_events"] = patient["adverse_events"]

    if "Adverse_Events" in patient:

        followup["Adverse_Events"] = patient["Adverse_Events"]

    return followup


# ============================================================
# 1. CLINICAL CHANGES
# ============================================================

@router.post(
    "/clinical-changes",
    response_model=Dict[str, Any]
)
def get_clinical_changes(req: AnalysisRequest):

    patient = _load_patient(req.patient_id)

    followup = _build_followup(
        patient,
        req.followup_overrides
    )

    try:

        changes = endpoint_service.calculate_clinical_changes(
            patient,
            followup
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Clinical change analysis failed: {str(exc)}"
        )

    return {

        "patient_id": req.patient_id,

        "baseline": {

            "HbA1c_percent":
                patient.get("HbA1c_percent"),

            "Weight_kg":
                patient.get("Weight_kg"),

            "Fasting_Glucose_mg_dL":
                patient.get("Fasting_Glucose_mg_dL"),

            "BMI":
                patient.get("BMI"),

            "BP_Systolic_mmHg":
                patient.get("BP_Systolic_mmHg"),
        },

        "followup": followup,

        "changes": changes,

        "disclaimer":
            "Follow-up values are simulated from population-level "
            "GLP-1 response data unless actual follow-up measurements "
            "are provided."
    }


# ============================================================
# 2. ENDPOINT EVALUATION
# ============================================================

@router.post(
    "/endpoints",
    response_model=Dict[str, Any]
)
def evaluate_endpoints(req: AnalysisRequest):

    patient = _load_patient(req.patient_id)

    followup = _build_followup(
        patient,
        req.followup_overrides
    )

    try:

        results = endpoint_service.evaluate_endpoint(
            patient,
            followup
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Endpoint evaluation failed: {str(exc)}"
        )

    return {

        "patient_id": req.patient_id,

        **results
    }


# ============================================================
# 3. SAFETY ANALYSIS
# ============================================================

@router.post(
    "/safety",
    response_model=Dict[str, Any]
)
def analyze_safety(req: AnalysisRequest):

    patient = _load_patient(req.patient_id)

    followup = _build_followup(
        patient,
        req.followup_overrides
    )

    try:

        safety = safety_service.analyze_adverse_events(
            followup
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Safety analysis failed: {str(exc)}"
        )

    return {

        "patient_id": req.patient_id,

        **safety
    }


# ============================================================
# 4. ML PREDICTION
# ============================================================

@router.post(
    "/ml-predict",
    response_model=Dict[str, Any]
)
def ml_predict(req: AnalysisRequest):

    patient = _load_patient(req.patient_id)

    try:

        prediction = ml_service.predict_patient(
            patient
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"ML prediction failed: {str(exc)}"
        )

    return {

        "patient_id": req.patient_id,

        **prediction,

        "model_metrics":
            ml_service.metadata.get(
                "metrics",
                {}
            )
            if ml_service.metadata
            else {}
    }


# ============================================================
# 5. SHAP EXPLANATION
# ============================================================

@router.post(
    "/shap-explain",
    response_model=Dict[str, Any]
)
def shap_explain(req: AnalysisRequest):

    patient = _load_patient(req.patient_id)

    try:

        contributions = shap_service.explain_patient(
            patient
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"SHAP explanation failed: {str(exc)}"
        )

    return {

        "patient_id": req.patient_id,

        "feature_contributions":
            contributions,

        "explanation_note":
            "Feature contributions are calculated using SHAP "
            "values. Positive contributions increase predicted "
            "risk and negative contributions decrease predicted risk."
    }


# ============================================================
# 6. FULL ANALYSIS
# ============================================================

@router.post(
    "/full-analysis",
    response_model=Dict[str, Any]
)
def full_analysis(req: AnalysisRequest):

    patient = _load_patient(req.patient_id)

    followup = _build_followup(
        patient,
        req.followup_overrides
    )

    # --------------------------------------------------------
    # Clinical changes
    # --------------------------------------------------------

    try:

        changes = endpoint_service.calculate_clinical_changes(
            patient,
            followup
        )

    except Exception as exc:

        changes = {
            "error": str(exc)
        }

    # --------------------------------------------------------
    # Trial endpoints
    # --------------------------------------------------------

    try:

        endpoints = endpoint_service.evaluate_endpoint(
            patient,
            followup
        )

    except Exception as exc:

        endpoints = {
            "error": str(exc)
        }

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    try:

        safety = safety_service.analyze_adverse_events(
            followup
        )

    except Exception as exc:

        safety = {
            "error": str(exc)
        }

    # --------------------------------------------------------
    # ML + SHAP
    # --------------------------------------------------------

    try:

        ml_prediction = ml_service.predict_patient(
            patient
        )

    except Exception as exc:

        ml_prediction = {

            "error": str(exc),

            "model_name": "N/A",

            "prediction": None,

            "probability": None
        }

    try:

        shap_contributions = shap_service.explain_patient(
            patient
        )

    except Exception as exc:

        shap_contributions = [

            {
                "error": str(exc)
            }

        ]

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {

        "patient_id": req.patient_id,

        "clinical_changes": changes,

        "endpoint_evaluation": endpoints,

        "safety_summary": safety,

        "ml_prediction": ml_prediction,

        "shap_contributions": shap_contributions,

        "analysis_status": "completed",

        "disclaimer":
            "This analysis combines deterministic eligibility "
            "rules, clinical endpoint evaluation, safety analysis, "
            "machine-learning prediction and SHAP-based "
            "explanation. Follow-up values may be simulated when "
            "actual measurements are unavailable. This system "
            "does not constitute a medical or regulatory assessment."
    }