from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd

router = APIRouter()


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PATIENT_CSV = PROJECT_ROOT / "synthetic_type2_diabetes_trial_volunteers_200.csv"


# ============================================================
# LOAD PATIENT DATASET
# ============================================================

def load_patients() -> pd.DataFrame:
    """
    Load the synthetic diabetes patient dataset.
    This function is also imported by matching_routes.py.
    """

    if not PATIENT_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Patient CSV not found: {PATIENT_CSV}"
        )

    try:
        df = pd.read_csv(PATIENT_CSV)

        # Clean column names
        df.columns = df.columns.astype(str).str.strip()

        if "Patient_ID" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Patient_ID column not found in CSV",
                    "available_columns": df.columns.tolist()
                }
            )

        # Normalize Patient_ID column
        df["Patient_ID"] = (
            df["Patient_ID"]
            .astype(str)
            .str.strip()
        )

        return df

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load patient CSV: {str(exc)}"
        )


# ============================================================
# NORMALIZE PATIENT ID
# ============================================================

def normalize_id(value: Any) -> str:
    """
    Makes these equivalent:

    DM001
    DM-001
    dm001
    dm-001
    DM_001
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
# LOAD SINGLE PATIENT
# ============================================================

def _load_patient(patient_id: str) -> Dict[str, Any]:

    df = load_patients()

    requested_id = normalize_id(patient_id)

    normalized_ids = df["Patient_ID"].apply(normalize_id)

    rows = df[normalized_ids == requested_id]

    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Patient '{patient_id}' not found",
                "requested_normalized_id": requested_id,
                "available_sample_ids": (
                    df["Patient_ID"]
                    .head(20)
                    .tolist()
                )
            }
        )

    patient_data = rows.iloc[0].to_dict()

    return {
        key: (
            None
            if pd.isna(value)
            else value
        )
        for key, value in patient_data.items()
    }


# ============================================================
# GET ALL PATIENTS
# ============================================================

@router.get("/", response_model=List[Dict[str, Any]])
def get_all_patients(limit: int = 200):

    df = load_patients()

    df_limited = df.head(limit)

    df_filled = df_limited.fillna("Unknown")

    return df_filled.to_dict(
        orient="records"
    )


# ============================================================
# GET PATIENT BY ID
# ============================================================

@router.get(
    "/{patient_id}",
    response_model=Dict[str, Any]
)
def get_patient_by_id(patient_id: str):

    return _load_patient(patient_id)