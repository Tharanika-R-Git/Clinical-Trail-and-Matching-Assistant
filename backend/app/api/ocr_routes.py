from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any

from backend.app.services.ocr_service import OCRService
from backend.app.services.patient_matching_service import PatientMatchingService

router = APIRouter()

# Max upload size (bytes): 10 MB.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/x-ms-bmp",
}

ocr_service = OCRService()


def _build_patient_from_fields(
    extracted_fields: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Collapse parsed fields into a patient-like record for the engine."""
    patient: Dict[str, Any] = {}

    for key, detail in extracted_fields.items():
        patient[key] = detail["value"]

    return patient


@router.post("/extract", response_model=Dict[str, Any])
async def extract_lab_report(file: UploadFile = File(...)):
    """
    OCR a lab report image, extract structured lab values and evaluate
    them against the trial eligibility criteria.
    """

    # --------------------------------------------------------
    # 1. Validate upload
    # --------------------------------------------------------

    content_type = (file.content_type or "").lower()

    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported file type. Upload a PNG, JPEG, WEBP, BMP "
                "or TIFF image."
            )
        )

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Uploaded file exceeds the 10 MB size limit."
        )

    # --------------------------------------------------------
    # 2. OCR
    # --------------------------------------------------------

    try:
        parsed = ocr_service.extract_and_parse(contents)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {type(exc).__name__}: {exc}"
        )

    extracted_fields = parsed["extracted_fields"]

    # --------------------------------------------------------
    # 3. Eligibility evaluation on extracted values
    # --------------------------------------------------------

    trial_data = {
        "trial_id": "NCT05502562",
        "condition": "Type 2 Diabetes",
        "intervention": {"name": "Oral semaglutide", "type": "drug"},
        "eligibility": {
            "inclusion": [
                "Age >= 18 years",
                "HbA1c between 7.0% and 10.5%",
                "Fasting plasma glucose < 270 mg/dL",
                "eGFR >= 60",
                "Patient must provide written informed consent",
            ],
            "exclusion": [
                "Pregnancy or breastfeeding",
                "History of pancreatitis",
                "Severe cardiovascular disease",
            ],
        },
    }

    patient = _build_patient_from_fields(extracted_fields)

    eligibility = None
    eligibility_note = None

    if patient:
        service = PatientMatchingService(trial_data)
        eligibility = service.evaluate_patient(patient)
        eligibility_note = (
            "Eligibility is computed from lab values extracted from the "
            "report only. Non-lab criteria (age, diagnosis, consent, "
            "pregnancy, medication) are marked UNKNOWN and require "
            "clinical review."
        )
    else:
        eligibility_note = (
            "No lab values were extracted from the image, so eligibility "
            "could not be evaluated."
        )

    # --------------------------------------------------------
    # 4. Response
    # --------------------------------------------------------

    return {
        "filename": file.filename or "image",
        "raw_text": parsed["raw_text"],
        "extracted_fields": extracted_fields,
        "unparsed_count": parsed["unparsed_count"],
        "eligibility": eligibility,
        "eligibility_note": eligibility_note,
    }
