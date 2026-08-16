"""
OCR service for lab reports.

Pipeline:
    1. Validate + preprocess the image (Pillow / OpenCV).
    2. Run RapidOCR to obtain raw recognized text.
    3. Parse the raw text into structured lab fields using a field
       registry (label aliases -> canonical column names).

The parsing logic is kept as a pure module-level function so it can be
unit-tested without the OCR engine (no model download required).
"""

import io
import re
from typing import Dict, List, Any, Optional

import numpy as np

# ============================================================
# FIELD REGISTRY
# ============================================================
#
# Each entry describes how to recognize a lab parameter inside the
# OCR text.  `aliases` are label strings that typically appear next
# to the numeric result.  `unit` is the expected unit of the raw
# result.  `convert` optionally converts from `unit` to the canonical
# project unit used by the CSV/eligibility engine.

FIELDS: List[Dict[str, Any]] = [
    {
        "key": "HbA1c_percent",
        "aliases": [
            "glycated haemoglobin",
            "glycated hemoglobin",
            "hba1c",
            "a1c",
            "hb a1c",
            "haemoglobin a1c",
            "hemoglobin a1c",
        ],
        "unit": "%",
        "convert": [
            {
                "pattern": r"mmol\s*/\s*mol",
                "from": "mmol/mol",
                "fn": lambda v: round((v + 23.5) / 10.93, 2),
            }
        ],
    },
    {
        "key": "Fasting_Glucose_mg_dL",
        "aliases": [
            "fasting blood sugar",
            "fasting plasma glucose",
            "fasting glucose",
            "fbs",
            "fpg",
            "blood glucose fasting",
            "glucose fasting",
        ],
        "unit": "mg/dL",
        "convert": [
            {
                "pattern": r"mmol\s*/\s*l",
                "from": "mmol/L",
                "fn": lambda v: round(v * 18.018, 1),
            }
        ],
    },
    {
        "key": "Postprandial_Glucose_mg_dL",
        "aliases": [
            "post prandial blood sugar",
            "postprandial blood sugar",
            "post prandial glucose",
            "postprandial glucose",
            "ppbs",
            "pp glucose",
        ],
        "unit": "mg/dL",
        "convert": [
            {
                "pattern": r"mmol\s*/\s*l",
                "from": "mmol/L",
                "fn": lambda v: round(v * 18.018, 1),
            }
        ],
    },
    {
        "key": "Creatinine_mg_dL",
        "aliases": ["serum creatinine", "creatinine", "s. creatinine"],
        "unit": "mg/dL",
        "convert": [
            {
                "pattern": r"\u00b5?mol\s*/\s*l",
                "from": "umol/L",
                "fn": lambda v: round(v / 88.4, 2),
            }
        ],
    },
    {
        "key": "Urea_mg_dL",
        "aliases": ["blood urea", "serum urea", "urea", "bun"],
        "unit": "mg/dL",
        "convert": [
            {
                "pattern": r"mmol\s*/\s*l",
                "from": "mmol/L",
                "fn": lambda v: round(v * 6.006, 1),
            }
        ],
    },
    {
        "key": "eGFR_mL_min_1_73m2",
        "aliases": [
            "egfr",
            "e-gfr",
            "estimated gfr",
            "glomerular filtration rate",
            "gfr",
        ],
        "unit": "mL/min/1.73m2",
    },
    {
        "key": "ALT_U_L",
        "aliases": ["alanine aminotransferase", "alat", "sgpt", "alt (sgpt)", "alt"],
        "unit": "U/L",
    },
    {
        "key": "AST_U_L",
        "aliases": ["aspartate aminotransferase", "asat", "sgot", "ast (sgot)", "ast"],
        "unit": "U/L",
    },
    {
        "key": "ALP_U_L",
        "aliases": ["alkaline phosphatase", "alp", "alk phos", "alkaline phosph"],
        "unit": "U/L",
    },
    {
        "key": "Bilirubin_mg_dL",
        "aliases": [
            "total bilirubin",
            "serum bilirubin",
            "bilirubin total",
            "bilirubin",
        ],
        "unit": "mg/dL",
        "convert": [
            {
                "pattern": r"mmol\s*/\s*l",
                "from": "mmol/L",
                "fn": lambda v: round(v / 17.1, 2),
            }
        ],
    },
    {
        "key": "Hemoglobin_g_dL",
        "aliases": ["haemoglobin", "hemoglobin", "hb", "hgb"],
        "unit": "g/dL",
    },
    {
        "key": "WBC_per_uL",
        "aliases": [
            "total leucocyte count",
            "total leukocyte count",
            "white blood cells",
            "white blood cell count",
            "wbc",
            "tlc",
        ],
        "unit": "/uL",
    },
    {
        "key": "RBC_million_per_uL",
        "aliases": [
            "red blood cell count",
            "red blood cells",
            "rbc count",
            "rbc",
            "erythrocyte count",
        ],
        "unit": "million/uL",
    },
    {
        "key": "Platelets_per_uL",
        "aliases": ["platelet count", "platelets", "plts", "thrombocyte"],
        "unit": "/uL",
    },
    {
        "key": "Total_Cholesterol_mg_dL",
        "aliases": ["total cholesterol", "serum cholesterol", "cholesterol total"],
        "unit": "mg/dL",
        "convert": [
            {
                "pattern": r"mmol\s*/\s*l",
                "from": "mmol/L",
                "fn": lambda v: round(v * 38.67, 1),
            }
        ],
    },
    {
        "key": "LDL_mg_dL",
        "aliases": [
            "ldl cholesterol",
            "ldl-c",
            "low density lipoprotein",
            "ldl",
        ],
        "unit": "mg/dL",
        "convert": [
            {
                "pattern": r"mmol\s*/\s*l",
                "from": "mmol/L",
                "fn": lambda v: round(v * 38.67, 1),
            }
        ],
    },
    {
        "key": "HDL_mg_dL",
        "aliases": [
            "hdl cholesterol",
            "hdl-c",
            "high density lipoprotein",
            "hdl",
        ],
        "unit": "mg/dL",
        "convert": [
            {
                "pattern": r"mmol\s*/\s*l",
                "from": "mmol/L",
                "fn": lambda v: round(v * 38.67, 1),
            }
        ],
    },
    {
        "key": "Triglycerides_mg_dL",
        "aliases": ["triglycerides", "triglyceride", "tgl", "serum triglycerides"],
        "unit": "mg/dL",
        "convert": [
            {
                "pattern": r"mmol\s*/\s*l",
                "from": "mmol/L",
                "fn": lambda v: round(v * 88.57, 1),
            }
        ],
    },
]

# Urine strip results use qualitative values.
URINE_FIELDS: List[Dict[str, Any]] = [
    {
        "key": "Urine_Glucose",
        "aliases": ["urine glucose", "urinary glucose", "sugar", "urine sugar"],
        "values": {
            "negative": "Negative",
            "neg": "Negative",
            "trace": "Trace",
            "1+": "1+",
            "2+": "2+",
            "3+": "3+",
            "4+": "4+",
        },
    },
    {
        "key": "Urine_Protein",
        "aliases": ["urine protein", "urinary protein", "protein", "albumin urine"],
        "values": {
            "negative": "Negative",
            "neg": "Negative",
            "trace": "Trace",
            "1+": "1+",
            "2+": "2+",
            "3+": "3+",
            "4+": "4+",
        },
    },
    {
        "key": "Urine_Ketones",
        "aliases": ["urine ketones", "urinary ketones", "ketones", "ketone bodies"],
        "values": {
            "negative": "Negative",
            "neg": "Negative",
            "trace": "Trace",
            "1+": "1+",
            "2+": "2+",
            "3+": "3+",
        },
    },
    {
        "key": "Urine_Albumin_Status",
        "aliases": [
            "urine albumin",
            "microalbumin",
            "albumin creatinine ratio",
            "uacr",
        ],
        "values": {
            "negative": "Negative",
            "neg": "Negative",
            "trace": "Trace",
            "microalbuminuria": "Microalbuminuria",
            "macroalbuminuria": "Macroalbuminuria",
        },
    },
]


def _normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, and unify OCR-friendly artifacts."""
    normalized = text.lower()
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    normalized = normalized.replace("\u00b7", ".").replace("*", "")
    normalized = re.sub(r"[|!]", "1", normalized)
    normalized = re.sub(r"[^\x00-\x7f]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized)


def _parse_number(value_text: str) -> Optional[float]:
    """Parse a numeric token that may contain commas or trailing junk."""
    cleaned = value_text.replace(",", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _apply_conversions(raw_value: float, unit_text: str, convert_rules: List[Dict]) -> float:
    """Convert a raw value to the canonical unit when a rule matches."""
    normalized_unit = unit_text.lower()
    for rule in convert_rules:
        if re.search(rule["pattern"], normalized_unit):
            return rule["fn"](raw_value)
    return raw_value


def _extract_quantitative_fields(normalized: str) -> Dict[str, Any]:
    """
    Match each quantitative field by scanning for its alias label
    followed by a nearby numeric token (with optional unit).
    """
    results: Dict[str, Any] = {}

    for field in FIELDS:
        key = field["key"]
        aliases = sorted(field["aliases"], key=len, reverse=True)
        matched = False

        for alias in aliases:
            if matched:
                break

            alias_re = re.escape(alias)

            # Prefer a value on the same line:  "<label> <number> <unit>"
            line_pattern = re.compile(
                rf"\b(?P<label>{alias_re})(?!\w)\s*[:\-.]?\s*"
                rf"(?P<num>\d[\d,\.]*)\s*"
                rf"(?P<unit>[a-zA-Z%/\u00b5\u03bc]*)\b"
            )

            for line_match in line_pattern.finditer(normalized):
                raw_number = _parse_number(line_match.group("num"))
                if raw_number is None:
                    continue

                unit_text = line_match.group("unit") or field.get("unit", "")

                # Guard against units leaking in (e.g. mmol/L for glucose).
                value = _apply_conversions(
                    raw_number, unit_text, field.get("convert", [])
                )

                results[key] = {
                    "value": value,
                    "unit": field["unit"],
                    "confidence": "HIGH",
                    "raw_match": line_match.group(0),
                }
                matched = True
                break

            if matched:
                break

        if matched:
            continue

        # Fallback: label and value separated across tokens (multi-line
        # OCR output) e.g. "HbA1c\n8.4 %".
        for alias in aliases:
            alias_re = re.escape(alias)
            span_pattern = re.compile(
                rf"\b(?P<label>{alias_re})(?!\w)[^0-9]{{0,40}}?"
                rf"(?P<num>\d[\d,\.]*)\s*"
                rf"(?P<unit>[a-zA-Z%/\u00b5\u03bc]*)\b"
            )

            for span_match in span_pattern.finditer(normalized):
                raw_number = _parse_number(span_match.group("num"))
                if raw_number is None:
                    continue

                unit_text = span_match.group("unit") or field.get("unit", "")

                # Avoid absurd values (OCR noise like dates or sample ids).
                if not (0.01 <= raw_number <= 100000):
                    continue

                value = _apply_conversions(
                    raw_number, unit_text, field.get("convert", [])
                )

                results[key] = {
                    "value": value,
                    "unit": field["unit"],
                    "confidence": "MEDIUM",
                    "raw_match": span_match.group(0),
                }
                break

    return results


def _extract_qualitative_fields(normalized: str) -> Dict[str, Any]:
    """Match qualitative (urine strip) fields by alias + result word."""
    results: Dict[str, Any] = {}

    for field in URINE_FIELDS:
        key = field["key"]
        aliases = sorted(field["aliases"], key=len, reverse=True)

        for alias in aliases:
            alias_re = re.escape(alias)

            pattern = re.compile(
                rf"\b(?P<label>{alias_re})(?!\w)\s*[:\-.]?\s*"
                rf"(?P<val>[a-z0-9+]+)"
            )

            for match in pattern.finditer(normalized):
                candidate = match.group("val").strip(".")
                value = field["values"].get(candidate)

                if value is not None:
                    results[key] = {
                        "value": value,
                        "unit": None,
                        "confidence": "HIGH",
                        "raw_match": match.group(0),
                    }
                    break

            if key in results:
                break

    return results


def parse_lab_report(raw_text: str) -> Dict[str, Any]:
    """
    Parse raw OCR text into structured lab fields.

    Returns:
        {
            "extracted_fields": { <field>: {value, unit, confidence, raw_match} },
            "unparsed_count": int,
        }
    """
    normalized = _normalize_text(raw_text)

    extracted = {}
    extracted.update(_extract_quantitative_fields(normalized))
    extracted.update(_extract_qualitative_fields(normalized))

    return {
        "extracted_fields": extracted,
        "unparsed_count": len(FIELDS) + len(URINE_FIELDS) - len(extracted),
    }


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def _box_center(box: Any) -> tuple:
    """
    Return the (x, y) center of a RapidOCR detection box.

    Handles two box shapes:
      - [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
      - [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    if isinstance(box, (list, tuple)) and len(box) == 4 and all(
        isinstance(point, (list, tuple)) and len(point) >= 2
        for point in box
    ):
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        return sum(xs) / 4.0, sum(ys) / 4.0

    if (
        isinstance(box, (list, tuple))
        and len(box) == 8
        and all(isinstance(value, (int, float)) for value in box)
    ):
        xs = box[0::2]
        ys = box[1::2]
        return sum(xs) / 4.0, sum(ys) / 4.0

    return 0.0, 0.0


def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode, validate and enhance an image for OCR."""
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception as exc:
        raise ValueError(
            f"Could not read image: {type(exc).__name__}: {exc}"
        )

    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")

    # Upscale small images so OCR sees more detail.
    target_min = 1600
    if min(image.size) < target_min:
        scale = target_min / min(image.size)
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.LANCZOS,
        )

    # RapidOCR performs its own contrast/threshold handling, so just
    # feed a grayscale array.
    return np.array(image.convert("L"))


# ============================================================
# OCR SERVICE
# ============================================================

class OCRService:
    """Thin wrapper around the RapidOCR engine with lazy loading."""

    _engine = None
    _engine_error = None

    def __init__(self, engine: Any = None):
        # `engine` can be injected for testing.
        self._injected_engine = engine

    def _get_engine(self) -> Any:
        if self._injected_engine is not None:
            return self._injected_engine

        if OCRService._engine is not None:
            return OCRService._engine

        if OCRService._engine_error is not None:
            raise OCRService._engine_error

        try:
            from rapidocr_onnxruntime import RapidOCR

            OCRService._engine = RapidOCR()
            return OCRService._engine
        except Exception as exc:
            OCRService._engine_error = RuntimeError(
                "OCR engine unavailable. Run `pip install -r requirements.txt` "
                "to install rapidocr_onnxruntime and its model files."
            )
            raise OCRService._engine_error from exc

    def extract_text(self, image_bytes: bytes) -> str:
        """
        Run OCR on the provided image bytes and return the recognized
        text, reconstructed into table rows (one line per row).

        Lab reports are typically tables: OCR returns each cell as a
        separate detection, so cells are grouped by their vertical
        position and ordered left-to-right to rebuild the original rows.
        """
        array = _preprocess_image(image_bytes)

        engine = self._get_engine()

        result = engine(array)

        # RapidOCR returns (recognitions, elapsed_times) as a tuple.
        # `recognitions` is a list of [box, text, confidence] triplets.
        if (
            isinstance(result, tuple)
            and len(result) >= 2
            and isinstance(result[0], list)
        ):
            result = result[0]

        detections = []
        height = array.shape[0] if hasattr(array, "shape") else 1600

        if isinstance(result, list):
            for item in result:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) >= 2
                    and isinstance(item[1], str)
                ):
                    cx, cy = _box_center(item[0])
                    detections.append({
                        "text": item[1].strip(),
                        "cx": cx,
                        "cy": cy,
                    })

        if not detections:
            return ""

        detections.sort(key=lambda d: (d["cy"], d["cx"]))

        # Group detections whose vertical centers are close into rows.
        tolerance = max(12.0, 0.02 * height)
        rows = []
        current_row = []
        anchor_y = None

        for detection in detections:
            if (
                anchor_y is None
                or abs(detection["cy"] - anchor_y) <= tolerance
            ):
                current_row.append(detection)
                if anchor_y is None:
                    anchor_y = detection["cy"]
            else:
                rows.append(current_row)
                current_row = [detection]
                anchor_y = detection["cy"]

        if current_row:
            rows.append(current_row)

        lines = []
        for row in rows:
            row.sort(key=lambda d: d["cx"])
            lines.append(" ".join(d["text"] for d in row if d["text"]))

        return "\n".join(lines)

    def extract_and_parse(self, image_bytes: bytes) -> Dict[str, Any]:
        raw_text = self.extract_text(image_bytes)
        parsed = parse_lab_report(raw_text)
        parsed["raw_text"] = raw_text
        return parsed
