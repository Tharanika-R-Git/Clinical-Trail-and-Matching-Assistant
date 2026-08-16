import os
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, Any
from backend.app.ml.train import train_and_evaluate, MODELS_DIR


class MLService:
    def __init__(self):
        self.preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
        self.model_path        = os.path.join(MODELS_DIR, "best_model.pkl")
        self.metadata_path     = os.path.join(MODELS_DIR, "model_metadata.json")
        self._load_resources()

    def _load_resources(self):
        if os.path.exists(self.preprocessor_path) and os.path.exists(self.model_path):
            self.preprocessor = joblib.load(self.preprocessor_path)
            self.model        = joblib.load(self.model_path)
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.preprocessor = None
            self.model        = None
            self.metadata     = None

    def ensure_trained(self, force_retrain: bool = False):
        if self.model is None or force_retrain:
            print("Model files not found or retrain forced. Running training pipeline...")
            self.metadata = train_and_evaluate()
            self._load_resources()
        return self.metadata

    # ------------------------------------------------------------------
    # Map synthetic volunteer schema → historical diabetic_data schema
    # ------------------------------------------------------------------
    def map_patient_to_features(self, patient: Dict[str, Any]) -> pd.DataFrame:
        current_meds = str(patient.get("Current_Medication", "")).lower()
        other_conds  = str(patient.get("Other_Medical_Conditions", "")).lower()

        # Number of concurrent medications
        if current_meds in ["diet and exercise only", "none documented", "no current medication", "none", ""]:
            num_meds = 0
        elif "+" in current_meds:
            num_meds = len(current_meds.split("+"))
        else:
            num_meds = 1

        # Number of comorbidities/diagnoses
        num_diagnoses = 2
        for cond in ["hypertension", "high cholesterol", "dyslipidaemia", "neuropathy", "retinopathy"]:
            if cond in other_conds:
                num_diagnoses += 1

        # Gender mapping
        gender = str(patient.get("Gender", "Female"))
        if gender not in ["Male", "Female"]:
            gender = "Female"

        # Age bin
        age_val    = float(patient.get("Age", 50))
        age_decade = int(age_val // 10) * 10
        age_bin    = f"[{age_decade}-{age_decade + 10})"

        # HbA1c → A1Cresult mapping
        hba1c = float(patient.get("HbA1c_percent", 7.0))
        if hba1c > 8.0:
            a1c_result = ">8"
        elif hba1c > 7.0:
            a1c_result = ">7"
        else:
            a1c_result = "Norm"

        # Glucose serum
        glucose = patient.get("Fasting_Glucose_mg_dL")
        if glucose is not None:
            g = float(glucose)
            max_glu_serum = ">300" if g >= 300 else (">200" if g >= 200 else "Norm")
        else:
            max_glu_serum = "None"

        # Medication flags
        metformin = "Steady" if "metformin" in current_meds else "No"
        insulin   = "Steady" if "insulin" in current_meds else "No"
        change    = "Ch" if num_meds > 1 else "No"
        diab_med  = "Yes" if num_meds > 0 else "No"

        # Derived engineered features (must match preprocess.py)
        on_insulin        = 1 if "insulin" in current_meds else 0
        primary_diab      = 1  # screening cohort — all T2DM
        total_prior_visits = 0
        had_prior_admission = 0
        num_med_changes   = max(0, num_meds - 1)  # approximation

        # Primary diagnosis category: diabetes for this cohort
        diag_cat = "diabetes"

        feature_dict = {
            # Numeric
            "time_in_hospital":          [3.0],
            "num_lab_procedures":        [45.0],
            "num_procedures":            [1.0],
            "num_medications":           [float(num_meds)],
            "number_outpatient":         [0.0],
            "number_emergency":          [0.0],
            "number_inpatient":          [0.0],
            "number_diagnoses":          [float(num_diagnoses)],
            "total_prior_visits":        [float(total_prior_visits)],
            "had_prior_admission":       [float(had_prior_admission)],
            "num_med_changes":           [float(num_med_changes)],
            "on_insulin":                [float(on_insulin)],
            "primary_diab":              [float(primary_diab)],
            "admission_type_id":         [1.0],   # Emergency = 1 (reasonable default)
            "discharge_disposition_id":  [1.0],   # Discharged home = 1
            "admission_source_id":       [7.0],   # Emergency Room = 7
            # Categorical
            "race":           ["Caucasian"],
            "gender":         [gender],
            "age":            [age_bin],
            "A1Cresult":      [a1c_result],
            "max_glu_serum":  [max_glu_serum],
            "metformin":      [metformin],
            "insulin":        [insulin],
            "change":         [change],
            "diabetesMed":    [diab_med],
            "diag_1_cat":     [diag_cat],
            "diag_2_cat":     ["other"],
            "diag_3_cat":     ["other"],
        }

        return pd.DataFrame(feature_dict)

    def predict_patient(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        self.ensure_trained()

        df_patient = self.map_patient_to_features(patient)

        # Align columns with what the preprocessor expects
        expected_num = self.metadata.get("num_cols", [])
        expected_cat = self.metadata.get("cat_cols", [])
        expected_cols = expected_num + expected_cat
        for col in expected_cols:
            if col not in df_patient.columns:
                df_patient[col] = 0 if col in expected_num else "Unknown"
        df_patient = df_patient[expected_cols]

        X_proc = self.preprocessor.transform(df_patient)
        prob   = float(self.model.predict_proba(X_proc)[0, 1])
        pred   = int(self.model.predict(X_proc)[0])

        return {
            "model_name":  self.metadata["best_model"],
            "prediction":  pred,
            "probability": round(prob, 4),
            "explanation_estimate": (
                "The ML model provides an analytical estimate of 30-day readmission risk "
                "based on the patient's clinical and medication profile."
            ),
        }
