from typing import Dict, Any, List, Tuple
import pandas as pd

class EligibilityRuleEngine:
    def __init__(self, trial_data: Dict[str, Any]):
        self.trial_id = trial_data.get("trial_id", "NCT05502562")
        self.inclusion_rules = trial_data.get("eligibility", {}).get("inclusion", [])
        self.exclusion_rules = trial_data.get("eligibility", {}).get("exclusion", [])

    def evaluate_patient(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single patient record against inclusion and exclusion criteria.
        Outputs: PASS, FAIL, UNKNOWN, MANUAL_REVIEW for each criterion.
        """
        criteria_results = []
        overall_status = "POTENTIALLY_ELIGIBLE"
        
        # Inclusion Checks
        # 1. Age >= 18
        age = patient.get("Age")
        if age is None or age == "Unknown":
            criteria_results.append({
                "criterion_id": "INC-001",
                "type": "inclusion",
                "field": "age",
                "status": "UNKNOWN",
                "source_text": "Age >= 18 years",
                "patient_value": "Missing",
                "evidence": "Age data not present in patient profile."
            })
        else:
            try:
                age_val = float(age)
                passed = age_val >= 18
                criteria_results.append({
                    "criterion_id": "INC-001",
                    "type": "inclusion",
                    "field": "age",
                    "status": "PASS" if passed else "FAIL",
                    "source_text": "Age >= 18 years",
                    "patient_value": f"{age_val} years",
                    "evidence": f"Patient age is {age_val} which is >= 18." if passed else f"Patient age is {age_val} which is < 18."
                })
            except Exception:
                criteria_results.append({
                    "criterion_id": "INC-001",
                    "type": "inclusion",
                    "field": "age",
                    "status": "MANUAL_REVIEW",
                    "source_text": "Age >= 18 years",
                    "patient_value": str(age),
                    "evidence": f"Age value '{age}' could not be parsed numerically."
                })

        # 2. HbA1c between 7.0 and 10.5
        hba1c = patient.get("HbA1c_percent")
        if hba1c is None or hba1c == "Unknown":
            criteria_results.append({
                "criterion_id": "INC-002",
                "type": "inclusion",
                "field": "hba1c",
                "status": "UNKNOWN",
                "source_text": "HbA1c between 7.0% and 10.5% at screening",
                "patient_value": "Missing",
                "evidence": "HbA1c data not present in patient profile."
            })
        else:
            try:
                hba1c_val = float(hba1c)
                passed = 7.0 <= hba1c_val <= 10.5
                criteria_results.append({
                    "criterion_id": "INC-002",
                    "type": "inclusion",
                    "field": "hba1c",
                    "status": "PASS" if passed else "FAIL",
                    "source_text": "HbA1c between 7.0% and 10.5% at screening",
                    "patient_value": f"{hba1c_val}%",
                    "evidence": f"Patient HbA1c is {hba1c_val}%, within required 7.0%-10.5% range." if passed else f"Patient HbA1c is {hba1c_val}%, outside required 7.0%-10.5% range."
                })
            except Exception:
                criteria_results.append({
                    "criterion_id": "INC-002",
                    "type": "inclusion",
                    "field": "hba1c",
                    "status": "MANUAL_REVIEW",
                    "source_text": "HbA1c between 7.0% and 10.5% at screening",
                    "patient_value": str(hba1c),
                    "evidence": f"HbA1c value '{hba1c}' could not be parsed."
                })

        # 3. Fasting glucose < 270 mg/dL
        fpg = patient.get("Fasting_Glucose_mg_dL")
        if fpg is None or fpg == "Unknown":
            criteria_results.append({
                "criterion_id": "INC-003",
                "type": "inclusion",
                "field": "fasting_glucose",
                "status": "UNKNOWN",
                "source_text": "Fasting plasma glucose < 270 mg/dL",
                "patient_value": "Missing",
                "evidence": "Fasting glucose data not present."
            })
        else:
            try:
                fpg_val = float(fpg)
                passed = fpg_val < 270
                criteria_results.append({
                    "criterion_id": "INC-003",
                    "type": "inclusion",
                    "field": "fasting_glucose",
                    "status": "PASS" if passed else "FAIL",
                    "source_text": "Fasting plasma glucose < 270 mg/dL",
                    "patient_value": f"{fpg_val} mg/dL",
                    "evidence": f"Patient fasting glucose is {fpg_val} mg/dL, which is < 270 mg/dL." if passed else f"Patient fasting glucose is {fpg_val} mg/dL, which is >= 270 mg/dL."
                })
            except Exception:
                criteria_results.append({
                    "criterion_id": "INC-003",
                    "type": "inclusion",
                    "field": "fasting_glucose",
                    "status": "MANUAL_REVIEW",
                    "source_text": "Fasting plasma glucose < 270 mg/dL",
                    "patient_value": str(fpg),
                    "evidence": f"Fasting glucose value '{fpg}' could not be parsed."
                })

        # 4. eGFR >= 60
        egfr = patient.get("eGFR_mL_min_1_73m2")
        if egfr is None or egfr == "Unknown":
            criteria_results.append({
                "criterion_id": "INC-004",
                "type": "inclusion",
                "field": "egfr",
                "status": "UNKNOWN",
                "source_text": "eGFR >= 60 mL/min/1.73m2",
                "patient_value": "Missing",
                "evidence": "eGFR kidney function measurement not present."
            })
        else:
            try:
                egfr_val = float(egfr)
                passed = egfr_val >= 60
                criteria_results.append({
                    "criterion_id": "INC-004",
                    "type": "inclusion",
                    "field": "egfr",
                    "status": "PASS" if passed else "FAIL",
                    "source_text": "eGFR >= 60 mL/min/1.73m2",
                    "patient_value": f"{egfr_val} mL/min/1.73m2",
                    "evidence": f"Patient eGFR is {egfr_val}, indicating adequate kidney function (>= 60)." if passed else f"Patient eGFR is {egfr_val}, indicating renal impairment (< 60)."
                })
            except Exception:
                criteria_results.append({
                    "criterion_id": "INC-004",
                    "type": "inclusion",
                    "field": "egfr",
                    "status": "MANUAL_REVIEW",
                    "source_text": "eGFR >= 60 mL/min/1.73m2",
                    "patient_value": str(egfr),
                    "evidence": f"eGFR value '{egfr}' could not be parsed."
                })

        # 5. Written informed consent (Consent_for_Trial)
        consent = patient.get("Consent_for_Trial", "No")
        passed = str(consent).strip().lower() == "yes"
        criteria_results.append({
            "criterion_id": "INC-005",
            "type": "inclusion",
            "field": "consent",
            "status": "PASS" if passed else "FAIL",
            "source_text": "Patient must provide written informed consent",
            "patient_value": str(consent),
            "evidence": "Patient consented to clinical trial participation." if passed else "Patient has not provided trial consent."
        })

        # Exclusion Checks
        # 1. Pregnancy/breastfeeding
        pregnancy = patient.get("Pregnancy", "No")
        # Let's map if Pregnancy is Yes, or other forms
        preg_val = str(pregnancy).strip().lower()
        excluded = preg_val in ["yes", "y", "true", "pregnant"]
        criteria_results.append({
            "criterion_id": "EXC-001",
            "type": "exclusion",
            "field": "pregnancy",
            "status": "FAIL" if excluded else "PASS",  # For exclusion criteria, PASS means patient is NOT excluded
            "source_text": "Pregnancy or breastfeeding",
            "patient_value": str(pregnancy),
            "evidence": "Patient is pregnant/breastfeeding (EXCLUDED)." if excluded else "Patient is not pregnant/breastfeeding."
        })

        # 2. History of pancreatitis or cardiovascular diseases in medical conditions
        medical_conds = str(patient.get("Other_Medical_Conditions", "")).lower()
        pancreatitis_excl = "pancreatitis" in medical_conds
        cvd_excl = any(x in medical_conds for x in ["cardiovascular disease", "heart failure", "stroke", "myocardial infarction"])
        
        criteria_results.append({
            "criterion_id": "EXC-002",
            "type": "exclusion",
            "field": "pancreatitis",
            "status": "FAIL" if pancreatitis_excl else "PASS",
            "source_text": "History of pancreatitis",
            "patient_value": patient.get("Other_Medical_Conditions", "None documented"),
            "evidence": "History of pancreatitis found (EXCLUDED)." if pancreatitis_excl else "No history of pancreatitis documented."
        })
        
        criteria_results.append({
            "criterion_id": "EXC-003",
            "type": "exclusion",
            "field": "cardiovascular_disease",
            "status": "FAIL" if cvd_excl else "PASS",
            "source_text": "Known cardiovascular disease, heart failure, or stroke in past 180 days",
            "patient_value": patient.get("Other_Medical_Conditions", "None documented"),
            "evidence": "Recent cardiovascular conditions noted in medical profile (EXCLUDED)." if cvd_excl else "No recent cardiovascular exclusions documented."
        })

        # Aggregate Results
        passed_count = sum(1 for c in criteria_results if c["status"] == "PASS")
        failed_count = sum(1 for c in criteria_results if c["status"] == "FAIL")
        unknown_count = sum(1 for c in criteria_results if c["status"] == "UNKNOWN")
        review_count = sum(1 for c in criteria_results if c["status"] == "MANUAL_REVIEW")

        # Rules logic:
        # If any inclusion is FAIL, or any exclusion is FAIL (meaning they match the exclusion), status is NOT_ELIGIBLE
        if failed_count > 0:
            overall_status = "NOT_ELIGIBLE"
        elif unknown_count > 0 or review_count > 0:
            overall_status = "POTENTIALLY_ELIGIBLE_WITH_REVIEW"
        else:
            overall_status = "POTENTIALLY_ELIGIBLE"

        return {
            "patient_id": patient.get("Patient_ID", "Unknown"),
            "eligibility_status": overall_status,
            "passed": passed_count,
            "failed": failed_count,
            "unknown": unknown_count,
            "manual_review": review_count,
            "criteria_detail": criteria_results
        }
