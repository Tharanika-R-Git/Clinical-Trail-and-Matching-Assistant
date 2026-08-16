"""
endpoint_service.py — Clinical Trial Endpoint Evaluation Service
================================================================
Purely deterministic Python mathematics — NO LLM calls.
Evaluates whether the primary/secondary endpoints of trial NCT05502562 are
achieved based on observed baseline vs follow-up measurements.
"""

from typing import Dict, Any, Optional


class EndpointService:
    """
    Evaluate clinical trial endpoints from paired baseline/follow-up measurements.

    The primary endpoint for NCT05502562 is:
      'Change in HbA1c from baseline to week 40'

    Secondary endpoints:
      - Change in body weight from baseline to week 40
      - Change in fasting plasma glucose from baseline to week 40
      - Proportion of patients achieving HbA1c < 7.0%

    All calculations are deterministic arithmetic — no AI/LLM inference.
    """

    # NCT05502562 endpoint definitions
    PRIMARY_ENDPOINT = {
        "name": "Change in HbA1c from baseline to week 40",
        "type": "primary",
        "field_baseline": "HbA1c_percent",
        "field_followup": "HbA1c_percent",
        "unit": "%",
        "achievement_threshold": -1.0,   # >= 1.0% absolute reduction considered meaningful
        "direction": "decrease",
    }

    SECONDARY_ENDPOINTS = [
        {
            "name": "Change in body weight from baseline to week 40",
            "type": "secondary",
            "field_baseline": "Weight_kg",
            "field_followup": "Weight_kg",
            "unit": "kg",
            "achievement_threshold": -2.0,
            "direction": "decrease",
        },
        {
            "name": "Change in fasting plasma glucose from baseline to week 40",
            "type": "secondary",
            "field_baseline": "Fasting_Glucose_mg_dL",
            "field_followup": "Fasting_Glucose_mg_dL",
            "unit": "mg/dL",
            "achievement_threshold": -20.0,
            "direction": "decrease",
        },
        {
            "name": "Proportion achieving HbA1c < 7.0%",
            "type": "secondary",
            "field_baseline": "HbA1c_percent",
            "field_followup": "HbA1c_percent",
            "unit": "%",
            "achievement_threshold": 7.0,   # absolute target value (not change)
            "direction": "target_below",
        },
    ]

    # ------------------------------------------------------------------
    # Core evaluation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Parse a value to float, returning None on failure."""
        if value is None:
            return None
        try:
            v = float(str(value).replace(",", "").strip())
            return None if (v != v) else v   # NaN guard
        except (ValueError, TypeError):
            return None

    def _evaluate_single(
        self,
        endpoint_def: Dict,
        baseline: Dict,
        followup: Dict,
    ) -> Dict:
        """
        Evaluate one endpoint definition against baseline and follow-up dicts.

        Returns
        -------
        dict with keys:
            endpoint_name, endpoint_type, baseline_value, followup_value,
            observed_change, unit, result, threshold_used, source
        """
        name = endpoint_def["name"]
        ep_type = endpoint_def.get("type", "primary")
        b_field = endpoint_def["field_baseline"]
        f_field = endpoint_def["field_followup"]
        unit = endpoint_def.get("unit", "")
        threshold = endpoint_def["achievement_threshold"]
        direction = endpoint_def["direction"]

        b_val = self._safe_float(baseline.get(b_field))
        f_val = self._safe_float(followup.get(f_field))

        # Handle missing data gracefully
        if b_val is None or f_val is None:
            missing = []
            if b_val is None:
                missing.append(f"baseline {b_field}")
            if f_val is None:
                missing.append(f"followup {f_field}")
            return {
                "endpoint_name": name,
                "endpoint_type": ep_type,
                "baseline_value": b_val,
                "followup_value": f_val,
                "observed_change": None,
                "unit": unit,
                "result": "UNKNOWN",
                "threshold_used": threshold,
                "note": f"Missing data: {', '.join(missing)}. Cannot evaluate endpoint.",
                "source": "endpoint_service (deterministic)",
            }

        change = round(f_val - b_val, 4)

        # Evaluate achievement based on direction
        if direction == "decrease":
            # Endpoint achieved if change <= threshold (e.g. change <= -1.0%)
            achieved = change <= threshold
        elif direction == "increase":
            achieved = change >= threshold
        elif direction == "target_below":
            # Endpoint achieved if follow-up value is below the threshold
            achieved = f_val < threshold
        else:
            achieved = False

        result = "ACHIEVED" if achieved else "NOT_ACHIEVED"

        return {
            "endpoint_name": name,
            "endpoint_type": ep_type,
            "baseline_value": b_val,
            "followup_value": f_val,
            "observed_change": change,
            "unit": unit,
            "result": result,
            "threshold_used": threshold,
            "source": "endpoint_service (deterministic arithmetic)",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_endpoint(
        self,
        baseline: Dict[str, Any],
        followup: Dict[str, Any],
        endpoint: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the primary endpoint (and optionally a custom endpoint).

        Parameters
        ----------
        baseline : dict with patient baseline measurements
        followup : dict with patient follow-up measurements
        endpoint : optional custom endpoint definition dict; if None, uses primary

        Returns
        -------
        dict with primary evaluation + secondary evaluations
        """
        ep_def = endpoint if endpoint else self.PRIMARY_ENDPOINT
        primary_result = self._evaluate_single(ep_def, baseline, followup)

        # Always evaluate the three secondary endpoints
        secondary_results = [
            self._evaluate_single(ep, baseline, followup)
            for ep in self.SECONDARY_ENDPOINTS
        ]

        return {
            "primary": primary_result,
            "secondary": secondary_results,
            "disclaimer": "Endpoint evaluation is purely deterministic arithmetic. "
                          "This does not constitute a medical or regulatory assessment.",
        }

    def calculate_clinical_changes(
        self,
        baseline: Dict[str, Any],
        followup: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute all key clinical change metrics between baseline and follow-up.

        Returns
        -------
        dict with hba1c_change, weight_change, glucose_change, bmi_change,
        and raw baseline/followup values for each metric.
        """
        def _change(field: str) -> Optional[float]:
            b = self._safe_float(baseline.get(field))
            f = self._safe_float(followup.get(field))
            if b is None or f is None:
                return None
            return round(f - b, 4)

        hba1c_change = _change("HbA1c_percent")
        weight_change = _change("Weight_kg")
        glucose_change = _change("Fasting_Glucose_mg_dL")
        bmi_change = _change("BMI")
        bp_systolic_change = _change("BP_Systolic_mmHg")

        return {
            "hba1c_change": {
                "baseline": self._safe_float(baseline.get("HbA1c_percent")),
                "followup": self._safe_float(followup.get("HbA1c_percent")),
                "change": hba1c_change,
                "unit": "%",
                "direction": "decrease is improvement",
            },
            "weight_change": {
                "baseline": self._safe_float(baseline.get("Weight_kg")),
                "followup": self._safe_float(followup.get("Weight_kg")),
                "change": weight_change,
                "unit": "kg",
                "direction": "decrease is improvement",
            },
            "glucose_change": {
                "baseline": self._safe_float(baseline.get("Fasting_Glucose_mg_dL")),
                "followup": self._safe_float(followup.get("Fasting_Glucose_mg_dL")),
                "change": glucose_change,
                "unit": "mg/dL",
                "direction": "decrease is improvement",
            },
            "bmi_change": {
                "baseline": self._safe_float(baseline.get("BMI")),
                "followup": self._safe_float(followup.get("BMI")),
                "change": bmi_change,
                "unit": "kg/m2",
                "direction": "decrease is improvement",
            },
            "bp_systolic_change": {
                "baseline": self._safe_float(baseline.get("BP_Systolic_mmHg")),
                "followup": self._safe_float(followup.get("BP_Systolic_mmHg")),
                "change": bp_systolic_change,
                "unit": "mmHg",
                "direction": "decrease is improvement",
            },
            "source": "endpoint_service (deterministic arithmetic — no LLM)",
        }
