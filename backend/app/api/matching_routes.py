from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd

from backend.app.api.trial_routes import get_trial_details
from backend.app.api.patient_routes import load_patients
from backend.app.services.patient_matching_service import PatientMatchingService

router = APIRouter()


class MatchingRequest(BaseModel):
    trial_id: str = "NCT05502562"
    patient_dataset: str = "synthetic_type2_diabetes_trial_volunteers_200.csv"
    weights: Optional[Dict[str, float]] = None


class SimpleMatchResponse(BaseModel):
    best_patient_id: str
    match_score: float
    status: str


@router.post("/run", response_model=Dict[str, Any])
def run_matching(req: MatchingRequest):

    try:
        # 1. Get trial
        trial = get_trial_details(req.trial_id)

        # Convert TrialSchema to dictionary
        if hasattr(trial, "model_dump"):
            trial_data = trial.model_dump()
        else:
            trial_data = trial.dict()

        print("\n========== TRIAL DATA ==========")
        print(trial_data)

        # Make sure eligibility exists
        if "eligibility" not in trial_data:
            raise HTTPException(
                status_code=500,
                detail="Trial data missing 'eligibility'"
            )

        # 2. Load patients
        df = load_patients()

        print(f"\nLoaded patients: {len(df)}")
        print("Columns:")
        print(df.columns.tolist())

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="Patient dataset is empty"
            )

        # 3. Convert DataFrame to list of dictionaries
        patients_list = []

        for _, row in df.iterrows():

            patient = row.to_dict()

            cleaned_patient = {
                key: None if pd.isna(value) else value
                for key, value in patient.items()
            }

            patients_list.append(cleaned_patient)

        # 4. Run matching engine
        matching_service = PatientMatchingService(trial_data)

        ranked = matching_service.match_and_rank(
            patients_list,
            req.weights
        )

        if not ranked:
            raise HTTPException(
                status_code=404,
                detail="No patients matched"
            )

        # 5. Best patient
        best_patient = ranked[0]

        print("\n========== BEST MATCH ==========")
        print(best_patient)

        return {
            "trial_id": req.trial_id,
            "total_patients_analyzed": len(patients_list),
            "best_patient_id": best_patient["patient_id"],
            "match_score": best_patient["match_score"],
            "status": best_patient["eligibility_status"],
            "candidates": ranked[:20]
        }

    except HTTPException:
        raise

    except Exception as e:

        print("\n========== MATCHING ERROR ==========")
        print(type(e).__name__)
        print(str(e))
        print("====================================\n")

        raise HTTPException(
            status_code=500,
            detail=f"Matching failed: {type(e).__name__}: {str(e)}"
        )


@router.post("/run_simple", response_model=SimpleMatchResponse)
def run_simple_matching(req: MatchingRequest):

    result = run_matching(req)

    return SimpleMatchResponse(
        best_patient_id=result["best_patient_id"],
        match_score=result["match_score"],
        status=result["status"]
    )