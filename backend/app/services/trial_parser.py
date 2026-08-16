"""
trial_parser.py — Clinical Trial Parser & Eligibility Extractor
================================================================
Extracts structured eligibility criteria, endpoints, and study design
from the NCT05502562 trial CSV or uses hardcoded trial data for the
PIONEER PLUS study.

All extraction is deterministic — no LLM calls.
"""

from typing import Dict, Any, List, Optional
import os
import pandas as pd


_NCT05502562_STRUCTURED: Dict[str, Any] = {
    "trial_id": "NCT05502562",
    "title": "PIONEER PLUS: A Phase 3b Trial of Oral Semaglutide in Adults with Type 2 Diabetes",
    "condition": "Type 2 Diabetes Mellitus",
    "phase": "Phase 3b",
    "intervention": {
        "name": "Oral semaglutide (Rybelsus/Ozempic)",
        "type": "drug",
        "mechanism": "GLP-1 receptor agonist",
        "doses": ["25 mg", "50 mg"],
        "frequency": "Once daily",
    },
    "eligibility": {
        "inclusion": [
            "Age >= 18 years",
            "Diagnosed with Type 2 Diabetes Mellitus",
            "HbA1c between 7.0% and 10.5% at screening",
            "Fasting plasma glucose < 270 mg/dL",
            "On stable metformin or lifestyle intervention for at least 8 weeks",
            "eGFR >= 60 mL/min/1.73m2",
            "Patient must provide written informed consent",
        ],
        "exclusion": [
            "Pregnancy or breastfeeding",
            "Type 1 Diabetes Mellitus",
            "History of pancreatitis",
            "Severe renal impairment (eGFR < 60 mL/min/1.73m2)",
            "Known cardiovascular disease, heart failure, or stroke in past 180 days",
            "Allergy or hypersensitivity to semaglutide or GLP-1 receptor agonists",
        ],
    },
    "outcomes": {
        "primary": ["Change in HbA1c from baseline to week 40"],
        "secondary": [
            "Change in body weight from baseline to week 40",
            "Change in fasting plasma glucose from baseline to week 40",
            "Proportion of patients achieving HbA1c < 7.0%",
        ],
    },
    "study_design": {
        "allocation": "Randomized",
        "intervention_model": "Parallel Assignment",
        "primary_purpose": "Treatment",
        "masking": "Double Blinded",
        "enrollment": 2000,
    },
    "timeframes": ["Baseline", "Week 12", "Week 24", "Week 40"],
    "eligibility_rules": [
        {"criterion_id": "INC-001", "type": "inclusion", "field": "Age", "description": "Age >= 18 years", "operator": ">=", "threshold": 18.0, "unit": "years"},
        {"criterion_id": "INC-002", "type": "inclusion", "field": "HbA1c_percent", "description": "HbA1c between 7.0% and 10.5% at screening", "operator": "between", "min_value": 7.0, "max_value": 10.5, "unit": "%"},
        {"criterion_id": "INC-003", "type": "inclusion", "field": "Fasting_Glucose_mg_dL", "description": "Fasting plasma glucose < 270 mg/dL", "operator": "<", "threshold": 270.0, "unit": "mg/dL"},
        {"criterion_id": "INC-004", "type": "inclusion", "field": "eGFR_mL_min_1_73m2", "description": "eGFR >= 60 mL/min/1.73m2", "operator": ">=", "threshold": 60.0, "unit": "mL/min/1.73m2"},
        {"criterion_id": "INC-005", "type": "inclusion", "field": "Consent_for_Trial", "description": "Written informed consent provided", "operator": "equals", "expected_value": "Yes"},
        {"criterion_id": "EXC-001", "type": "exclusion", "field": "Pregnancy", "description": "Pregnancy or breastfeeding", "operator": "not_equals", "expected_value": "Yes"},
        {"criterion_id": "EXC-002", "type": "exclusion", "field": "Other_Medical_Conditions", "description": "No history of pancreatitis", "operator": "not_contains", "keyword": "pancreatitis"},
        {"criterion_id": "EXC-003", "type": "exclusion", "field": "Other_Medical_Conditions", "description": "No known cardiovascular disease", "operator": "not_contains_any", "keywords": ["cardiovascular disease", "heart failure", "stroke", "myocardial infarction"]},
    ],
    "source": "trial_parser (deterministic — no LLM)",
}


class TrialParser:
    SUPPORTED_TRIALS = {"NCT05502562"}

    def parse(self, trial_id: str, csv_path: Optional[str] = None) -> Dict[str, Any]:
        if trial_id not in self.SUPPORTED_TRIALS:
            raise ValueError(f"Trial '{trial_id}' not supported. Supported trials: {self.SUPPORTED_TRIALS}")
        result = dict(_NCT05502562_STRUCTURED)
        if csv_path and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    row = df.iloc[0]
                    if "Study Title" in row and pd.notna(row["Study Title"]):
                        result["title"] = str(row["Study Title"])
                    if "Conditions" in row and pd.notna(row["Conditions"]):
                        result["condition"] = str(row["Conditions"])
                    result["csv_enriched"] = True
            except Exception as exc:
                result["csv_enriched"] = False
                result["csv_error"] = str(exc)
        return result

    def get_eligibility_rules(self, trial_id: str) -> List[Dict[str, Any]]:
        return self.parse(trial_id).get("eligibility_rules", [])

    def get_endpoints(self, trial_id: str) -> Dict[str, Any]:
        return self.parse(trial_id).get("outcomes", {})

    def get_inclusion_criteria(self, trial_id: str) -> List[str]:
        return self.parse(trial_id).get("eligibility", {}).get("inclusion", [])

    def get_exclusion_criteria(self, trial_id: str) -> List[str]:
        return self.parse(trial_id).get("eligibility", {}).get("exclusion", [])

    def get_timeframes(self, trial_id: str) -> List[str]:
        return self.parse(trial_id).get("timeframes", [])
