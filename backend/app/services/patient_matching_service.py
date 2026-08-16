from typing import List, Dict, Any, Optional
import math


class PatientMatchingService:
    """
    Patient-Trial Matching Engine

    Performs:
    1. Eligibility evaluation
    2. Clinical similarity scoring
    3. Demographic compatibility
    4. Semantic relevance
    5. Data completeness
    6. Final ranking
    """

    def __init__(self, trial_data: Dict[str, Any]):
        # Support both Pydantic model_dump() output and normal dictionaries
        if hasattr(trial_data, "model_dump"):
            trial_data = trial_data.model_dump()
        elif hasattr(trial_data, "dict"):
            trial_data = trial_data.dict()

        self.trial_data = trial_data or {}

    # ---------------------------------------------------------
    # Utility functions
    # ---------------------------------------------------------

    def _safe_float(self, value, default=None):
        try:
            if value is None:
                return default

            if isinstance(value, str):
                value = value.strip()

            number = float(value)

            if math.isnan(number):
                return default

            return number

        except (ValueError, TypeError):
            return default

    def _text(self, value):
        if value is None:
            return ""

        return str(value).strip().lower()

    def _get_value(self, patient, *fields, default=None):
        """
        Returns the first available patient field.
        Allows different CSV column naming conventions.
        """

        for field in fields:
            if field in patient:
                value = patient.get(field)

                if value is not None:
                    text = str(value).strip().lower()

                    if text not in ["", "nan", "none", "unknown"]:
                        return value

        return default

    # ---------------------------------------------------------
    # Eligibility Evaluation
    # ---------------------------------------------------------

    def evaluate_patient(self, patient: Dict[str, Any]) -> Dict[str, Any]:

        criteria_detail = []

        passed = 0
        failed = 0
        unknown = 0

        # ---------------------------------------------
        # 1. Age >= 18
        # ---------------------------------------------

        age = self._safe_float(
            self._get_value(patient, "Age", "age")
        )

        if age is None:
            status = "UNKNOWN"
            unknown += 1
        elif age >= 18:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        criteria_detail.append({
            "criterion": "Age >= 18 years",
            "type": "inclusion",
            "status": status,
            "patient_value": age
        })

        # ---------------------------------------------
        # 2. Type 2 Diabetes
        # ---------------------------------------------

        diabetes_type = self._text(
            self._get_value(
                patient,
                "Diabetes_Type",
                "diabetes_type",
                "Condition"
            )
        )

        if not diabetes_type:
            status = "UNKNOWN"
            unknown += 1
        elif (
            "type 2" in diabetes_type
            or "type2" in diabetes_type
            or "t2dm" in diabetes_type
        ):
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        criteria_detail.append({
            "criterion": "Diagnosed with Type 2 Diabetes Mellitus",
            "type": "inclusion",
            "status": status,
            "patient_value": diabetes_type
        })

        # ---------------------------------------------
        # 3. HbA1c 7.0 - 10.5
        # ---------------------------------------------

        hba1c = self._safe_float(
            self._get_value(
                patient,
                "HbA1c_percent",
                "HbA1c",
                "HBA1C"
            )
        )

        if hba1c is None:
            status = "UNKNOWN"
            unknown += 1
        elif 7.0 <= hba1c <= 10.5:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        criteria_detail.append({
            "criterion": "HbA1c between 7.0% and 10.5%",
            "type": "inclusion",
            "status": status,
            "patient_value": hba1c
        })

        # ---------------------------------------------
        # 4. Fasting glucose < 270
        # ---------------------------------------------

        glucose = self._safe_float(
            self._get_value(
                patient,
                "Fasting_Glucose_mg_dL",
                "Fasting_Glucose",
                "Glucose"
            )
        )

        if glucose is None:
            status = "UNKNOWN"
            unknown += 1
        elif glucose < 270:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        criteria_detail.append({
            "criterion": "Fasting plasma glucose < 270 mg/dL",
            "type": "inclusion",
            "status": status,
            "patient_value": glucose
        })

        # ---------------------------------------------
        # 5. Stable metformin/lifestyle
        # ---------------------------------------------

        medication = self._text(
            self._get_value(
                patient,
                "Current_Medication",
                "Medication",
                "CurrentMedication"
            )
        )

        treatment_duration = self._safe_float(
            self._get_value(
                patient,
                "Treatment_Duration_Weeks",
                "Medication_Duration_Weeks",
                "Stable_Treatment_Weeks"
            )
        )

        if medication:
            if "metformin" in medication:
                status = "PASS"
                passed += 1
            elif "lifestyle" in medication:
                status = "PASS"
                passed += 1
            else:
                status = "UNKNOWN"
                unknown += 1
        else:
            status = "UNKNOWN"
            unknown += 1

        criteria_detail.append({
            "criterion": "Stable metformin or lifestyle intervention",
            "type": "inclusion",
            "status": status,
            "patient_value": medication,
            "duration_weeks": treatment_duration
        })

        # ---------------------------------------------
        # 6. eGFR >= 60
        # ---------------------------------------------

        egfr = self._safe_float(
            self._get_value(
                patient,
                "eGFR_mL_min_1_73m2",
                "eGFR",
                "EGFR"
            )
        )

        if egfr is None:
            status = "UNKNOWN"
            unknown += 1
        elif egfr >= 60:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        criteria_detail.append({
            "criterion": "eGFR >= 60 mL/min/1.73m2",
            "type": "inclusion",
            "status": status,
            "patient_value": egfr
        })

        # ---------------------------------------------
        # 7. Consent
        # ---------------------------------------------

        consent = self._text(
            self._get_value(
                patient,
                "Consent_for_Trial",
                "Consent",
                "Informed_Consent"
            )
        )

        if consent in ["yes", "true", "1", "y", "consented"]:
            status = "PASS"
            passed += 1
        elif consent in ["no", "false", "0", "n"]:
            status = "FAIL"
            failed += 1
        else:
            status = "UNKNOWN"
            unknown += 1

        criteria_detail.append({
            "criterion": "Written informed consent",
            "type": "inclusion",
            "status": status,
            "patient_value": consent
        })

        # =================================================
        # EXCLUSION CRITERIA
        # =================================================

        # Pregnancy / breastfeeding
        pregnancy = self._text(
            self._get_value(
                patient,
                "Pregnancy",
                "Pregnant",
                "Pregnancy_Status"
            )
        )

        breastfeeding = self._text(
            self._get_value(
                patient,
                "Breastfeeding",
                "Breastfeeding_Status"
            )
        )

        if pregnancy in ["yes", "true", "1"] or breastfeeding in ["yes", "true", "1"]:
            status = "FAIL"
            failed += 1
        elif pregnancy or breastfeeding:
            status = "PASS"
            passed += 1
        else:
            status = "UNKNOWN"
            unknown += 1

        criteria_detail.append({
            "criterion": "No pregnancy or breastfeeding",
            "type": "exclusion",
            "status": status,
            "patient_value": {
                "pregnancy": pregnancy,
                "breastfeeding": breastfeeding
            }
        })

        # Type 1 diabetes
        if diabetes_type:
            if "type 1" in diabetes_type or "type1" in diabetes_type:
                status = "FAIL"
                failed += 1
            else:
                status = "PASS"
                passed += 1
        else:
            status = "UNKNOWN"
            unknown += 1

        criteria_detail.append({
            "criterion": "No Type 1 Diabetes Mellitus",
            "type": "exclusion",
            "status": status
        })

        # Pancreatitis
        pancreatitis = self._text(
            self._get_value(
                patient,
                "History_of_Pancreatitis",
                "Pancreatitis",
                "Pancreatitis_History"
            )
        )

        if pancreatitis in ["yes", "true", "1"]:
            status = "FAIL"
            failed += 1
        elif pancreatitis in ["no", "false", "0"]:
            status = "PASS"
            passed += 1
        else:
            status = "UNKNOWN"
            unknown += 1

        criteria_detail.append({
            "criterion": "No history of pancreatitis",
            "type": "exclusion",
            "status": status,
            "patient_value": pancreatitis
        })

        # Cardiovascular disease
        cardiovascular = self._text(
            self._get_value(
                patient,
                "Cardiovascular_Disease",
                "CVD",
                "Heart_Disease"
            )
        )

        if cardiovascular in ["yes", "true", "1"]:
            status = "FAIL"
            failed += 1
        elif cardiovascular in ["no", "false", "0"]:
            status = "PASS"
            passed += 1
        else:
            status = "UNKNOWN"
            unknown += 1

        criteria_detail.append({
            "criterion": "No recent cardiovascular disease/heart failure/stroke",
            "type": "exclusion",
            "status": status,
            "patient_value": cardiovascular
        })

        # Semaglutide / GLP-1 allergy
        allergy = self._text(
            self._get_value(
                patient,
                "Semaglutide_Allergy",
                "GLP1_Allergy",
                "Drug_Allergy"
            )
        )

        if allergy in ["yes", "true", "1"]:
            status = "FAIL"
            failed += 1
        elif allergy in ["no", "false", "0"]:
            status = "PASS"
            passed += 1
        else:
            status = "UNKNOWN"
            unknown += 1

        criteria_detail.append({
            "criterion": "No allergy to semaglutide or GLP-1 receptor agonists",
            "type": "exclusion",
            "status": status,
            "patient_value": allergy
        })

        # ---------------------------------------------
        # Final eligibility status
        # ---------------------------------------------

        if failed > 0:
            eligibility_status = "NOT_ELIGIBLE"

        elif unknown > 0:
            eligibility_status = "POTENTIALLY_ELIGIBLE_WITH_REVIEW"

        else:
            eligibility_status = "POTENTIALLY_ELIGIBLE"

        return {
            "eligibility_status": eligibility_status,
            "passed": passed,
            "failed": failed,
            "unknown": unknown,
            "manual_review": unknown > 0,
            "criteria_detail": criteria_detail
        }

    # =========================================================
    # MATCH + RANK
    # =========================================================

    def match_and_rank(
        self,
        patients: List[Dict[str, Any]],
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:

        if weights is None:
            weights = {
                "eligibility": 0.60,
                "clinical": 0.15,
                "demographic": 0.10,
                "semantic": 0.10,
                "completeness": 0.05
            }

        ranked_patients = []

        for patient in patients:

            # ---------------------------------------------
            # 1. Eligibility
            # ---------------------------------------------

            elig_res = self.evaluate_patient(patient)

            total_criteria = len(elig_res["criteria_detail"])

            eligibility_compatibility = (
                elig_res["passed"] / total_criteria
                if total_criteria > 0
                else 0.0
            )

            # ---------------------------------------------
            # 2. Clinical similarity
            # ---------------------------------------------

            diabetes_type = self._text(
                patient.get("Diabetes_Type")
            )

            medication = self._text(
                patient.get("Current_Medication")
            )

            hba1c = self._safe_float(
                patient.get("HbA1c_percent")
            )

            has_t2dm = (
                1.0
                if (
                    "type 2" in diabetes_type
                    or "type2" in diabetes_type
                    or "t2dm" in diabetes_type
                )
                else 0.0
            )

            on_metformin = (
                1.0
                if "metformin" in medication
                else 0.0
            )

            hba1c_good = (
                1.0
                if hba1c is not None and 7.0 <= hba1c <= 10.5
                else 0.0
            )

            clinical_similarity = (
                has_t2dm +
                on_metformin +
                hba1c_good
            ) / 3.0

            # ---------------------------------------------
            # 3. Demographic compatibility
            # ---------------------------------------------

            age = self._safe_float(
                patient.get("Age")
            )

            if age is None:
                demographic_compatibility = 0.0

            elif age >= 18:
                demographic_compatibility = 1.0

            else:
                demographic_compatibility = 0.0

            # ---------------------------------------------
            # 4. Semantic relevance
            # ---------------------------------------------

            text_bag = " ".join([
                self._text(patient.get("Symptoms")),
                self._text(patient.get("Other_Medical_Conditions")),
                self._text(patient.get("Trial_Criteria_Summary")),
                self._text(patient.get("Diabetes_Type")),
                self._text(patient.get("Current_Medication"))
            ])

            keywords = [
                "diabetes",
                "glucose",
                "hba1c",
                "metformin",
                "semaglutide",
                "insulin",
                "obesity",
                "weight"
            ]

            matches = sum(
                1 for keyword in keywords
                if keyword in text_bag
            )

            semantic_relevance = matches / len(keywords)

            # ---------------------------------------------
            # 5. Data completeness
            # ---------------------------------------------

            key_fields = [
                "Age",
                "Gender",
                "Height_cm",
                "Weight_kg",
                "BMI",
                "Blood_Pressure_mmHg",
                "Diabetes_Type",
                "Diagnosis_Duration_Years",
                "Fasting_Glucose_mg_dL",
                "HbA1c_percent",
                "Creatinine_mg_dL",
                "eGFR_mL_min_1_73m2",
                "ALT_U_L",
                "AST_U_L",
                "Consent_for_Trial"
            ]

            valid_fields = 0

            for field in key_fields:
                value = patient.get(field)

                if value is not None:
                    text = str(value).strip().lower()

                    if text not in [
                        "",
                        "unknown",
                        "nan",
                        "none"
                    ]:
                        valid_fields += 1

            data_completeness = (
                valid_fields / len(key_fields)
            )

            # ---------------------------------------------
            # 6. Final score
            # ---------------------------------------------

            match_score = (
                weights.get("eligibility", 0.60)
                * eligibility_compatibility

                + weights.get("clinical", 0.15)
                * clinical_similarity

                + weights.get("demographic", 0.10)
                * demographic_compatibility

                + weights.get("semantic", 0.10)
                * semantic_relevance

                + weights.get("completeness", 0.05)
                * data_completeness
            ) * 100

            # ---------------------------------------------
            # Result
            # ---------------------------------------------

            ranked_patients.append({
                "patient_id": patient.get("Patient_ID"),

                "eligibility_status":
                    elig_res["eligibility_status"],

                "match_score":
                    round(match_score, 1),

                "passed":
                    elig_res["passed"],

                "failed":
                    elig_res["failed"],

                "unknown":
                    elig_res["unknown"],

                "manual_review":
                    elig_res["manual_review"],

                "criteria_detail":
                    elig_res["criteria_detail"],

                "demographics": {
                    "age": patient.get("Age"),
                    "gender": patient.get("Gender"),
                    "location": patient.get("Location")
                },

                "clinical_metrics": {
                    "hba1c":
                        patient.get("HbA1c_percent"),

                    "fasting_glucose":
                        patient.get("Fasting_Glucose_mg_dL"),

                    "weight":
                        patient.get("Weight_kg"),

                    "bmi":
                        patient.get("BMI"),

                    "egfr":
                        patient.get("eGFR_mL_min_1_73m2"),

                    "current_medication":
                        patient.get("Current_Medication")
                }
            })

        # ---------------------------------------------
        # Ranking
        # ---------------------------------------------

        status_rank = {
            "POTENTIALLY_ELIGIBLE": 3,
            "POTENTIALLY_ELIGIBLE_WITH_REVIEW": 2,
            "NOT_ELIGIBLE": 1
        }

        ranked_patients.sort(
            key=lambda item: (
                status_rank.get(
                    item["eligibility_status"],
                    0
                ),
                item["match_score"]
            ),
            reverse=True
        )

        return ranked_patients