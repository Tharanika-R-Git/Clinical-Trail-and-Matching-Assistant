from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd

router = APIRouter()


# ============================================================
# PROJECT PATH
# ============================================================

# trial_routes.py
#     ↓ parents[0] = api
#     ↓ parents[1] = app
#     ↓ parents[2] = backend
#     ↓ parents[3] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# NCT05502562.csv is located directly in the project root
TRIALS_DIR = PROJECT_ROOT


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class InterventionSchema(BaseModel):
    name: str
    type: str


class EligibilitySchema(BaseModel):
    inclusion: List[str]
    exclusion: List[str]


class TrialSchema(BaseModel):
    trial_id: str
    condition: str
    intervention: InterventionSchema
    eligibility: EligibilitySchema
    outcomes: Dict[str, List[str]]
    study_design: Dict[str, Any]
    timeframes: List[str]


# ============================================================
# GET TRIAL DETAILS
# ============================================================

@router.get("/{trial_id}", response_model=TrialSchema)
def get_trial_details(trial_id: str):
    """
    Get structured clinical trial information.

    Example:
        GET /api/trials/NCT05502562
    """

    csv_path = TRIALS_DIR / f"{trial_id}.csv"

    print(f"[TRIAL] Looking for file: {csv_path}")

    # --------------------------------------------------------
    # Check whether CSV exists
    # --------------------------------------------------------

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Trial CSV file not found: {csv_path}"
        )

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    try:
        df = pd.read_csv(csv_path)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read trial CSV: {str(e)}"
        )

    # --------------------------------------------------------
    # Check empty CSV
    # --------------------------------------------------------

    if df.empty:
        raise HTTPException(
            status_code=500,
            detail=f"Trial CSV is empty: {csv_path}"
        )

    # --------------------------------------------------------
    # Debug information
    # --------------------------------------------------------

    print(f"[TRIAL] CSV loaded successfully")
    print(f"[TRIAL] Rows: {len(df)}")
    print(f"[TRIAL] Columns: {list(df.columns)}")

    # --------------------------------------------------------
    # Get first row
    # --------------------------------------------------------

    row = df.iloc[0]

    # --------------------------------------------------------
    # Helper function
    # --------------------------------------------------------

    def get_value(column_name: str, default: str = ""):
        """
        Safely retrieve a value from the CSV.
        Handles missing columns and NaN values.
        """

        if column_name not in df.columns:
            return default

        value = row[column_name]

        if pd.isna(value):
            return default

        return str(value).strip()

    # --------------------------------------------------------
    # Trial basic information
    # --------------------------------------------------------

    condition = get_value(
        "Conditions",
        "Type 2 Diabetes Mellitus"
    )

    intervention_raw = get_value(
        "Interventions",
        "DRUG: Oral semaglutide"
    )

    # --------------------------------------------------------
    # Parse intervention
    # --------------------------------------------------------

    intervention_type = "drug"
    intervention_name = "Oral semaglutide"

    if intervention_raw:

        if ":" in intervention_raw:

            parts = intervention_raw.split(":", 1)

            intervention_type = parts[0].strip().lower()
            intervention_name = parts[1].strip()

        else:

            intervention_name = intervention_raw

    # --------------------------------------------------------
    # Trial eligibility
    # --------------------------------------------------------

    inclusion_criteria = [
        "Age >= 18 years",
        "Diagnosed with Type 2 Diabetes Mellitus",
        "HbA1c between 7.0% and 10.5% at screening",
        "Fasting plasma glucose < 270 mg/dL",
        "On stable metformin or lifestyle intervention for at least 8 weeks",
        "eGFR >= 60 mL/min/1.73m2",
        "Patient must provide written informed consent"
    ]

    exclusion_criteria = [
        "Pregnancy or breastfeeding",
        "Type 1 Diabetes Mellitus",
        "History of pancreatitis",
        "Severe renal impairment (eGFR < 60 mL/min/1.73m2)",
        "Known cardiovascular disease, heart failure, or stroke in past 180 days",
        "Allergy or hypersensitivity to semaglutide or GLP-1 receptor agonists"
    ]

    # --------------------------------------------------------
    # Outcomes
    # --------------------------------------------------------

    outcomes = {
        "primary": [
            "Change in HbA1c from baseline to week 40"
        ],
        "secondary": [
            "Change in body weight from baseline to week 40",
            "Change in fasting plasma glucose from baseline to week 40",
            "Proportion of patients achieving HbA1c < 7.0%"
        ]
    }

    # --------------------------------------------------------
    # Study design
    # --------------------------------------------------------

    study_design = {
        "allocation": "Randomized",
        "intervention_model": "Parallel Assignment",
        "primary_purpose": "Treatment",
        "masking": "Double Blinded"
    }

    # --------------------------------------------------------
    # Timeframes
    # --------------------------------------------------------

    timeframes = [
        "Baseline",
        "Week 12",
        "Week 24",
        "Week 40"
    ]

    # --------------------------------------------------------
    # Build response
    # --------------------------------------------------------

    trial_data = TrialSchema(
        trial_id=trial_id,
        condition=condition,

        intervention=InterventionSchema(
            name=intervention_name,
            type=intervention_type
        ),

        eligibility=EligibilitySchema(
            inclusion=inclusion_criteria,
            exclusion=exclusion_criteria
        ),

        outcomes=outcomes,

        study_design=study_design,

        timeframes=timeframes
    )

    return trial_data


# ============================================================
# UPLOAD / LOAD TRIAL
# ============================================================

@router.post("/upload", response_model=TrialSchema)
def upload_trial(trial_id: str = "NCT05502562"):
    """
    Load a trial CSV and return structured trial information.

    This endpoint currently uses the CSV already present
    in the project root.
    """

    return get_trial_details(trial_id)