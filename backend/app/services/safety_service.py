"""
safety_service.py — Adverse Event Safety Analysis Service
==========================================================
Parses and counts adverse events from follow-up data.
All causality statements follow a fixed, disclaimer-based approach — no LLM.

IMPORTANT: This service reports events; it does NOT establish medical causality.
"""

from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Severity mapping — used to classify seriousness
# ---------------------------------------------------------------------------

_SERIOUS_KEYWORDS = {
    "pancreatitis",
    "myocardial infarction",
    "stroke",
    "heart failure",
    "cardiac arrest",
    "renal failure",
    "hepatic failure",
    "anaphylaxis",
    "severe hypoglycaemia",
    "hypoglycemic coma",
    "diabetic ketoacidosis",
    "hospitalisation",
    "hospitalization",
    "death",
    "serious",
    "sae",
    "severe",
    "life-threatening",
}

_MANDATORY_CAUSALITY_NOTE = (
    "Causality not established. Events reported during follow-up period. "
    "Clinical assessment required."
)


class SafetyService:
    """
    Parses adverse event data from a follow-up record and produces a
    structured safety summary.

    Methods
    -------
    analyze_adverse_events(followup_data) -> dict
        Main entry point. Returns structured safety summary.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ae_list(followup_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract adverse events from follow-up data dict.

        Accepts multiple common formats:
          - 'adverse_events' key holding a list of dicts or strings
          - 'Adverse_Events' (alternate casing)
          - fallback: check other safety-related string fields
        """
        ae_raw = (
            followup_data.get("adverse_events")
            or followup_data.get("Adverse_Events")
            or followup_data.get("adverse_events_list")
            or []
        )

        parsed: List[Dict[str, Any]] = []

        for item in ae_raw:
            if isinstance(item, dict):
                # Already structured — normalise keys
                ae_entry = {
                    "event": str(item.get("event", item.get("name", item.get("ae", "Unknown event")))).strip(),
                    "severity": str(item.get("severity", item.get("grade", "Not specified"))).strip(),
                    "onset_week": item.get("onset_week", item.get("week", None)),
                    "resolved": bool(item.get("resolved", item.get("resolution", False))),
                    "serious": False,  # computed below
                }
            elif isinstance(item, str):
                ae_entry = {
                    "event": item.strip(),
                    "severity": "Not specified",
                    "onset_week": None,
                    "resolved": False,
                    "serious": False,
                }
            else:
                continue

            # Classify seriousness by keyword matching
            event_lower = ae_entry["event"].lower()
            severity_lower = ae_entry["severity"].lower()
            ae_entry["serious"] = any(
                kw in event_lower or kw in severity_lower
                for kw in _SERIOUS_KEYWORDS
            )
            parsed.append(ae_entry)

        return parsed

    @staticmethod
    def _classify_severity(event_name: str, severity_str: str) -> str:
        """Standardise severity label."""
        s = severity_str.lower()
        if any(k in s for k in ("severe", "serious", "grade 3", "grade 4", "grade 5")):
            return "Severe"
        if any(k in s for k in ("moderate", "grade 2")):
            return "Moderate"
        if any(k in s for k in ("mild", "grade 1")):
            return "Mild"
        # Infer from event name keywords
        ev = event_name.lower()
        if any(k in ev for k in _SERIOUS_KEYWORDS):
            return "Severe"
        if any(k in ev for k in ("nausea", "vomiting", "diarrhoea", "diarrhea", "headache", "fatigue")):
            return "Mild"
        return "Unknown"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_adverse_events(self, followup_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse adverse events from follow-up data and return a structured safety summary.

        Parameters
        ----------
        followup_data : dict containing follow-up patient data including an
                        'adverse_events' key (list of dicts or strings)

        Returns
        -------
        dict with keys:
            adverse_events    : list of structured AE records
            total_count       : int
            serious_count     : int
            mild_moderate_count : int
            causality_note    : str (fixed disclaimer, never inferred)
            summary           : human-readable summary string
            source            : provenance note
        """
        ae_list = self._parse_ae_list(followup_data)

        # Enrich with standardised severity
        for ae in ae_list:
            ae["severity_standardised"] = self._classify_severity(
                ae["event"], ae["severity"]
            )

        total_count = len(ae_list)
        serious_count = sum(1 for ae in ae_list if ae["serious"])
        mild_moderate_count = total_count - serious_count

        # Build brief textual summary
        if total_count == 0:
            summary = "No adverse events were recorded in the follow-up data provided."
        else:
            serious_events = [ae["event"] for ae in ae_list if ae["serious"]]
            non_serious = [ae["event"] for ae in ae_list if not ae["serious"]]
            parts = []
            if non_serious:
                parts.append(f"{len(non_serious)} non-serious event(s): {', '.join(non_serious[:5])}")
            if serious_events:
                parts.append(f"{serious_count} serious event(s): {', '.join(serious_events[:5])}")
            summary = "; ".join(parts) + "."

        return {
            "adverse_events": ae_list,
            "total_count": total_count,
            "serious_count": serious_count,
            "mild_moderate_count": mild_moderate_count,
            "causality_note": _MANDATORY_CAUSALITY_NOTE,
            "summary": summary,
            "source": "safety_service — event counting only, no causality inference",
        }
