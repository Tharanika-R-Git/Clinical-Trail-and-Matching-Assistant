"""
test_ml_pipeline.py — ML Pipeline Verification Tests
=====================================================
Verifies target definitions, feature preprocessing, leak prevention,
model training metrics, and inference on synthetic patient data.
"""

import sys
import os
sys.path.insert(0, 'F:/PEC_Hack')

import pytest
import numpy as np
import pandas as pd

from backend.app.ml.preprocess import prepare_data
from backend.app.services.ml_service import MLService


DIABETIC_CSV = "F:/PEC_Hack/diabetic_data.csv"
SKIP_IF_NO_DATA = pytest.mark.skipif(
    not os.path.exists(DIABETIC_CSV),
    reason="diabetic_data.csv not found — skipping full ML pipeline tests"
)


# ---------------------------------------------------------------------------
# MLService — Map patient features
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ml_service():
    return MLService()


@pytest.fixture
def volunteer_patient():
    return {
        "Patient_ID": "DM-ML-001",
        "Age": 55,
        "Gender": "Female",
        "HbA1c_percent": 8.6,
        "Fasting_Glucose_mg_dL": 165,
        "Weight_kg": 88,
        "BMI": 31.2,
        "eGFR_mL_min_1_73m2": 75,
        "Current_Medication": "Metformin + Sitagliptin",
        "Other_Medical_Conditions": "Hypertension",
        "Diabetes_Type": "Type 2 Diabetes",
    }


class TestFeatureMapping:
    def test_map_patient_returns_dataframe(self, ml_service, volunteer_patient):
        df = ml_service.map_patient_to_features(volunteer_patient)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_required_columns_present(self, ml_service, volunteer_patient):
        df = ml_service.map_patient_to_features(volunteer_patient)
        required_cols = [
            "time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_outpatient", "number_emergency",
            "number_inpatient", "number_diagnoses",
            "race", "gender", "age", "A1Cresult", "metformin", "insulin",
            "change", "diabetesMed",
        ]
        for col in required_cols:
            assert col in df.columns, f"Column '{col}' missing from mapped features"

    def test_hba1c_binning_greater_than_8(self, ml_service):
        patient = {"HbA1c_percent": 8.9, "Age": 50, "Gender": "Male", "Current_Medication": "Metformin"}
        df = ml_service.map_patient_to_features(patient)
        assert df.iloc[0]["A1Cresult"] == ">8"

    def test_hba1c_binning_greater_than_7(self, ml_service):
        patient = {"HbA1c_percent": 7.6, "Age": 50, "Gender": "Male", "Current_Medication": "None"}
        df = ml_service.map_patient_to_features(patient)
        assert df.iloc[0]["A1Cresult"] == ">7"

    def test_hba1c_binning_normal(self, ml_service):
        patient = {"HbA1c_percent": 6.5, "Age": 50, "Gender": "Male", "Current_Medication": "None"}
        df = ml_service.map_patient_to_features(patient)
        assert df.iloc[0]["A1Cresult"] == "Norm"

    def test_metformin_detected(self, ml_service):
        patient = {"HbA1c_percent": 7.5, "Age": 50, "Gender": "Female", "Current_Medication": "Metformin"}
        df = ml_service.map_patient_to_features(patient)
        assert df.iloc[0]["metformin"] == "Steady"

    def test_no_medication_patient(self, ml_service):
        patient = {"HbA1c_percent": 7.5, "Age": 50, "Gender": "Female", "Current_Medication": "diet and exercise only"}
        df = ml_service.map_patient_to_features(patient)
        assert df.iloc[0]["num_medications"] == 0
        assert df.iloc[0]["diabetesMed"] == "No"

    def test_age_binning(self, ml_service):
        patient = {"HbA1c_percent": 7.5, "Age": 55, "Gender": "Female", "Current_Medication": "Metformin"}
        df = ml_service.map_patient_to_features(patient)
        assert df.iloc[0]["age"] == "[50-60)"

    def test_gender_default_for_unknown(self, ml_service):
        patient = {"HbA1c_percent": 7.5, "Age": 50, "Gender": "Prefer not to say", "Current_Medication": "Metformin"}
        df = ml_service.map_patient_to_features(patient)
        assert df.iloc[0]["gender"] in ["Male", "Female"]

    def test_multi_medication_change_flag(self, ml_service):
        patient = {"HbA1c_percent": 8.0, "Age": 60, "Gender": "Male", "Current_Medication": "Metformin + Insulin"}
        df = ml_service.map_patient_to_features(patient)
        assert df.iloc[0]["change"] == "Ch"


# ---------------------------------------------------------------------------
# Preprocessing pipeline — target definition and feature extraction
# ---------------------------------------------------------------------------

class TestPreprocessing:
    @SKIP_IF_NO_DATA
    def test_prepare_data_returns_splits(self):
        X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols = prepare_data(
            DIABETIC_CSV, sample_size=500
        )
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    @SKIP_IF_NO_DATA
    def test_target_is_binary(self):
        X_train, X_test, y_train, y_test, _, _, _ = prepare_data(DIABETIC_CSV, sample_size=500)
        unique_vals = set(y_train.unique()) | set(y_test.unique())
        assert unique_vals.issubset({0, 1}), f"Non-binary target values found: {unique_vals}"

    @SKIP_IF_NO_DATA
    def test_no_leakage_columns(self):
        X_train, _, _, _, _, _, _ = prepare_data(DIABETIC_CSV, sample_size=500)
        leakage_cols = ["patient_nbr", "encounter_id", "readmitted"]
        for col in leakage_cols:
            assert col not in X_train.columns, f"Leakage column '{col}' found in features"

    @SKIP_IF_NO_DATA
    def test_numeric_columns_present(self):
        X_train, _, _, _, _, num_cols, _ = prepare_data(DIABETIC_CSV, sample_size=500)
        for col in num_cols:
            assert col in X_train.columns

    @SKIP_IF_NO_DATA
    def test_categorical_columns_present(self):
        X_train, _, _, _, _, _, cat_cols = prepare_data(DIABETIC_CSV, sample_size=500)
        for col in cat_cols:
            assert col in X_train.columns

    @SKIP_IF_NO_DATA
    def test_preprocessor_transforms_without_error(self):
        X_train, X_test, y_train, _, preprocessor, _, _ = prepare_data(DIABETIC_CSV, sample_size=500)
        X_proc = preprocessor.fit_transform(X_train)
        assert X_proc.shape[0] == len(X_train)
        assert X_proc.shape[1] > 0

    @SKIP_IF_NO_DATA
    def test_no_nan_after_preprocessing(self):
        X_train, _, y_train, _, preprocessor, _, _ = prepare_data(DIABETIC_CSV, sample_size=500)
        X_proc = preprocessor.fit_transform(X_train)
        assert not np.any(np.isnan(X_proc)), "NaN values found after preprocessing"


# ---------------------------------------------------------------------------
# ML Model inference
# ---------------------------------------------------------------------------

class TestMLInference:
    def test_predict_returns_dict(self, ml_service, volunteer_patient):
        result = ml_service.predict_patient(volunteer_patient)
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, ml_service, volunteer_patient):
        result = ml_service.predict_patient(volunteer_patient)
        for key in ["model_name", "prediction", "probability"]:
            assert key in result, f"Key '{key}' missing from prediction result"

    def test_predict_probability_in_range(self, ml_service, volunteer_patient):
        result = ml_service.predict_patient(volunteer_patient)
        prob = result["probability"]
        assert 0.0 <= prob <= 1.0, f"Probability {prob} out of [0, 1] range"

    def test_predict_binary_prediction(self, ml_service, volunteer_patient):
        result = ml_service.predict_patient(volunteer_patient)
        assert result["prediction"] in [0, 1], f"Prediction {result['prediction']} not binary"

    def test_predict_deterministic(self, ml_service, volunteer_patient):
        r1 = ml_service.predict_patient(volunteer_patient)
        r2 = ml_service.predict_patient(volunteer_patient)
        assert r1["probability"] == r2["probability"], "Prediction is not deterministic"
        assert r1["prediction"] == r2["prediction"]

    def test_metadata_has_model_metrics(self, ml_service, volunteer_patient):
        ml_service.predict_patient(volunteer_patient)  # ensure trained
        assert ml_service.metadata is not None
        assert "metrics" in ml_service.metadata
        assert "best_model" in ml_service.metadata
        # Check at least one model's ROC-AUC exists
        metrics = ml_service.metadata["metrics"]
        assert len(metrics) > 0
        first_model_metrics = list(metrics.values())[0]
        assert "roc_auc" in first_model_metrics
        assert first_model_metrics["roc_auc"] > 0.5, "ROC-AUC below 0.5 suggests model failure"
