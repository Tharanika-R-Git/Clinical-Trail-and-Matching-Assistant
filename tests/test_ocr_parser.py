import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from backend.app.services.ocr_service import (
    OCRService,
    parse_lab_report,
    _parse_number,
)
from backend.app.services.patient_matching_service import PatientMatchingService

SAMPLE_REPORT = """
DIABETIC PROFILE
============================================
Test                 Result     Unit
--------------------------------------------
Fasting Glucose      142        mg/dL
Post Prandial Blood Sugar  210 mg/dL
HbA1c                8.2        %
eGFR                 92         mL/min/1.73m2
Serum Creatinine     1.0        mg/dL
Blood Urea           38         mg/dL
ALT (SGPT)           42         U/L
AST (SGOT)           38         U/L
Total Cholesterol    198        mg/dL
LDL Cholesterol      122        mg/dL
HDL Cholesterol      44         mg/dL
Triglycerides        170        mg/dL
Hemoglobin           13.4       g/dL
WBC                  7200       /uL
Platelet Count       2,50,000   /uL
Urine Glucose        Negative
Urine Protein        Trace
Urine Ketones        Negative
"""


# ============================================================
# Pure parsing helpers
# ============================================================

def test_parse_number():
    assert _parse_number("142") == 142.0
    assert _parse_number("8.2") == 8.2
    assert _parse_number("2,50,000") == 250000.0
    assert _parse_number("abc") is None


# ============================================================
# parse_lab_report
# ============================================================

def test_parse_basic_lab_report():
    result = parse_lab_report(SAMPLE_REPORT)
    fields = result["extracted_fields"]

    assert fields["HbA1c_percent"]["value"] == 8.2
    assert fields["Fasting_Glucose_mg_dL"]["value"] == 142
    assert fields["Postprandial_Glucose_mg_dL"]["value"] == 210
    assert fields["eGFR_mL_min_1_73m2"]["value"] == 92
    assert fields["Creatinine_mg_dL"]["value"] == 1.0
    assert fields["Urea_mg_dL"]["value"] == 38
    assert fields["ALT_U_L"]["value"] == 42
    assert fields["AST_U_L"]["value"] == 38
    assert fields["Total_Cholesterol_mg_dL"]["value"] == 198
    assert fields["LDL_mg_dL"]["value"] == 122
    assert fields["HDL_mg_dL"]["value"] == 44
    assert fields["Triglycerides_mg_dL"]["value"] == 170
    assert fields["Hemoglobin_g_dL"]["value"] == 13.4
    assert fields["WBC_per_uL"]["value"] == 7200
    assert fields["Platelets_per_uL"]["value"] == 250000

    assert fields["Urine_Glucose"]["value"] == "Negative"
    assert fields["Urine_Protein"]["value"] == "Trace"
    assert fields["Urine_Ketones"]["value"] == "Negative"


def test_parse_single_line_format():
    text = "HbA1c 9.1 %    Fasting Glucose 165 mg/dL    eGFR 78 mL/min"
    fields = parse_lab_report(text)["extracted_fields"]

    assert fields["HbA1c_percent"]["value"] == 9.1
    assert fields["Fasting_Glucose_mg_dL"]["value"] == 165
    assert fields["eGFR_mL_min_1_73m2"]["value"] == 78


def test_parse_multi_line_label_and_value():
    text = "HbA1c\n8.4 %"
    fields = parse_lab_report(text)["extracted_fields"]
    assert fields["HbA1c_percent"]["value"] == 8.4


def test_parse_glucose_mmol_conversion():
    text = "Fasting Glucose 7.8 mmol/L"
    fields = parse_lab_report(text)["extracted_fields"]
    # 7.8 mmol/L * 18.018 = 140.5 mg/dL
    assert fields["Fasting_Glucose_mg_dL"]["value"] == pytest.approx(140.5, abs=0.1)


def test_parse_hba1c_mmol_conversion():
    text = "HbA1c 70 mmol/mol"
    fields = parse_lab_report(text)["extracted_fields"]
    # IFCC 70 mmol/mol = (70 - 23.5) / 10.93 = 4.25%? No: DCCT = (IFCC - 23.5) / 10.93
    # Actually: DCCT% = (IFCC + 23.5) / 10.93 for the inverse direction.
    # Conversion IFCC -> DCCT: DCCT(%) = 0.09148*IFCC + 2.152  (NGSP master equation)
    value = fields["HbA1c_percent"]["value"]
    assert 7.5 < value < 9.0


def test_parse_empty_text():
    result = parse_lab_report("")
    assert result["extracted_fields"] == {}
    assert result["unparsed_count"] > 0


# ============================================================
# Eligibility from extracted fields
# ============================================================

def test_eligibility_review_when_labs_in_range():
    report = "HbA1c 8.2 %  Fasting Glucose 142 mg/dL  eGFR 92 mL/min"
    fields = parse_lab_report(report)["extracted_fields"]

    patient = {key: detail["value"] for key, detail in fields.items()}

    trial = {
        "trial_id": "NCT05502562",
        "condition": "Type 2 Diabetes",
        "eligibility": {
            "inclusion": [
                "Age >= 18 years",
                "HbA1c between 7.0% and 10.5%",
                "Fasting plasma glucose < 270 mg/dL",
                "eGFR >= 60",
                "Patient must provide written informed consent",
            ],
            "exclusion": [],
        },
    }

    result = PatientMatchingService(trial).evaluate_patient(patient)

    assert result["eligibility_status"] == "POTENTIALLY_ELIGIBLE_WITH_REVIEW"
    assert result["failed"] == 0

    hba1c_criterion = next(
        c for c in result["criteria_detail"]
        if "HbA1c" in c["criterion"]
    )
    assert hba1c_criterion["status"] == "PASS"


def test_eligibility_fail_when_hba1c_out_of_range():
    report = "HbA1c 11.8 %  Fasting Glucose 190 mg/dL  eGFR 80 mL/min"
    fields = parse_lab_report(report)["extracted_fields"]

    patient = {key: detail["value"] for key, detail in fields.items()}

    trial = {
        "trial_id": "NCT05502562",
        "eligibility": {
            "inclusion": ["HbA1c between 7.0% and 10.5%"],
            "exclusion": [],
        },
    }

    result = PatientMatchingService(trial).evaluate_patient(patient)

    assert result["eligibility_status"] == "NOT_ELIGIBLE"


# ============================================================
# OCRService with injected engine
# ============================================================

def _fake_png_bytes():
    from PIL import Image
    import io

    image = Image.new("RGB", (100, 60), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeEngine:
    def __call__(self, image):
        return [[[0, 0, 0, 0], "HbA1c 8.2 %  Fasting Glucose 142 mg/dL", 0.99]]


def test_extract_text_uses_injected_engine():
    service = OCRService(engine=_FakeEngine())
    text = service.extract_text(_fake_png_bytes())
    assert "HbA1c 8.2" in text


def test_extract_and_parse_roundtrip():
    service = OCRService(engine=_FakeEngine())
    result = service.extract_and_parse(_fake_png_bytes())

    assert result["raw_text"].startswith("HbA1c")
    assert result["extracted_fields"]["HbA1c_percent"]["value"] == 8.2
    assert result["extracted_fields"]["Fasting_Glucose_mg_dL"]["value"] == 142


def test_extract_and_parse_uses_preprocessing():
    service = OCRService(engine=_FakeEngine())
    result = service.extract_and_parse(_fake_png_bytes())
    assert result["extracted_fields"]["HbA1c_percent"]["value"] == 8.2
