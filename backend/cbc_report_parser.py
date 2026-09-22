# ============================================================
# AAROGYAM AI
# CBC VISUAL REPORT EXTRACTOR + UNIT NORMALIZER
# ============================================================
#
# PURPOSE:
#   1. Read a CBC PDF/image using Gemini.
#   2. Extract the patient's CBC RESULT and the unit shown
#      on the report.
#   3. Normalize different laboratory unit formats into the
#      exact units expected by the CBC ML model.
#   4. Flag ambiguous/unrecognized values for manual review.
#
# IMPORTANT:
#   This file ONLY extracts and normalizes laboratory values.
#
#   It does NOT:
#   - diagnose disease
#   - predict dengue/malaria
#   - calculate severity
#   - interpret abnormal results
#   - run XGBoost/LightGBM
#
# FLOW:
#
#   PDF / IMAGE
#        ↓
#   Gemini Vision
#        ↓
#   raw value + raw unit
#        ↓
#   unit normalization
#        ↓
#   canonical CBC values
#        ↓
#   doctor review
#        ↓
#   /api/cbc/assess
#
# ============================================================

import json
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field


# ============================================================
# ENVIRONMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured.\n"
        "Add it to CBC_Aarogyam/.env"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# CANONICAL UNITS
# ============================================================
#
# THESE ARE THE UNITS EXPECTED BY YOUR ML MODEL.
# ============================================================

CANONICAL_UNITS = {
    "Hb": "g/dL",
    "RBC": "million/µL",
    "WBC": "/µL",
    "Platelets": "/µL",
    "Neutrophils": "%",
    "Lymphocytes": "%",
    "Monocytes": "%",
    "Eosinophils": "%",
    "Basophils": "%",
    "MCV": "fL",
    "MCH": "pg",
    "MCHC": "g/dL",
    "RDW": "%",
}

CBC_FIELDS = list(CANONICAL_UNITS.keys())


# ============================================================
# GEMINI EXTRACTION SCHEMA
# ============================================================

class CBCMeasurement(BaseModel):
    value: Optional[float] = Field(
        default=None,
        description=(
            "Numeric patient result exactly as shown on "
            "the laboratory report."
        ),
    )

    unit: Optional[str] = Field(
        default=None,
        description=(
            "Unit exactly as shown on the report. "
            "Examples: g/dL, gm%, g/L, million/cumm, "
            "/cumm, /uL, /µL, lakh/cumm, "
            "10^3/uL, x10^9/L, %, fL, pg."
        ),
    )


class CBCExtraction(BaseModel):
    Hb: CBCMeasurement = Field(default_factory=CBCMeasurement)
    RBC: CBCMeasurement = Field(default_factory=CBCMeasurement)
    WBC: CBCMeasurement = Field(default_factory=CBCMeasurement)
    Platelets: CBCMeasurement = Field(
        default_factory=CBCMeasurement
    )

    Neutrophils: CBCMeasurement = Field(
        default_factory=CBCMeasurement
    )
    Lymphocytes: CBCMeasurement = Field(
        default_factory=CBCMeasurement
    )
    Monocytes: CBCMeasurement = Field(
        default_factory=CBCMeasurement
    )
    Eosinophils: CBCMeasurement = Field(
        default_factory=CBCMeasurement
    )
    Basophils: CBCMeasurement = Field(
        default_factory=CBCMeasurement
    )

    MCV: CBCMeasurement = Field(
        default_factory=CBCMeasurement
    )
    MCH: CBCMeasurement = Field(
        default_factory=CBCMeasurement
    )
    MCHC: CBCMeasurement = Field(
        default_factory=CBCMeasurement
    )
    RDW: CBCMeasurement = Field(
        default_factory=CBCMeasurement
    )


# ============================================================
# GEMINI EXTRACTION PROMPT
# ============================================================

EXTRACTION_PROMPT = """
You are a laboratory report DATA EXTRACTION assistant.

Your ONLY task is to read the uploaded CBC report and extract
the patient's actual CBC result values.

DO NOT:
- diagnose the patient
- identify diseases
- interpret abnormalities
- calculate severity
- calculate derived values
- infer missing values
- invent values

For every CBC parameter:

1. Find the patient's RESULT value.
2. Return the numeric value exactly as printed.
3. Return the UNIT exactly as printed.
4. If the value is absent or uncertain, return null.

IMPORTANT:
Return the patient's result, NOT the reference range.

Examples:

If the report says:
Platelets 1.52 lakh/cumm

return:

value = 1.52
unit = "lakh/cumm"

Do NOT convert it yourself.

If the report says:
WBC 6200 /cumm

return:

value = 6200
unit = "/cumm"

If the report says:
Hemoglobin 9.0 gm%

return:

value = 9.0
unit = "gm%"

If the report says:
RBC 3.13 Million/cumm

return:

value = 3.13
unit = "Million/cumm"

FIELD ALIASES:

Hb:
Hb, HGB, Hemoglobin, Haemoglobin

RBC:
RBC, RBC Count, Red Blood Cell Count,
Red Blood Cells, Erythrocytes

WBC:
WBC, WBC Count, White Blood Cell Count,
White Blood Cells, Leukocytes, TLC,
Total Leukocyte Count

Platelets:
Platelets, Platelet, Platelet Count, PLT

Neutrophils:
Neutrophils, Neutrophil, Neutrophils %,
Neutrophil %, Neut, NEU

Lymphocytes:
Lymphocytes, Lymphocyte, Lymphocytes %,
Lymphocyte %, Lymph, LYM

Monocytes:
Monocytes, Monocyte, Monocytes %,
Monocyte %, Mono, MONO

Eosinophils:
Eosinophils, Eosinophil, Eosinophils %,
Eosinophil %, Eos, EOS

Basophils:
Basophils, Basophil, Basophils %,
Basophil %, Baso, BASO

MCV:
MCV, Mean Corpuscular Volume

MCH:
MCH, Mean Corpuscular Hemoglobin,
Mean Corpuscular Haemoglobin

MCHC:
MCHC, Mean Corpuscular Hemoglobin Concentration,
Mean Corpuscular Haemoglobin Concentration

RDW:
RDW, RDW-CV, Red Cell Distribution Width

Do not guess a unit if it is not visible.
"""


# ============================================================
# FILE VALIDATION
# ============================================================

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
}


def validate_file(file_path: str) -> Path:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. "
            "Use PDF, PNG, JPG, or JPEG."
        )

    return path


# ============================================================
# UNIT TEXT NORMALIZATION
# ============================================================
#
# Converts many visually different representations into
# consistent internal unit strings.
#
# Examples:
#
# gm%             -> g/dl
# Million/cumm    -> million/µl
# /cumm           -> /µl
# lakh/cumm       -> lakh/µl
# x 10^3/µL       -> x10^3/µl
# ============================================================

def normalize_unit(unit: Optional[str]) -> str:

    if unit is None:
        return ""

    u = unit.strip().lower()

    # --------------------------------------------------------
    # Unicode / punctuation
    # --------------------------------------------------------

    u = u.replace("μ", "µ")
    u = u.replace("×", "x")
    u = u.replace("−", "-")

    # --------------------------------------------------------
    # Remove spaces
    # --------------------------------------------------------

    u = " ".join(u.split())
    u = u.replace(" ", "")

    # --------------------------------------------------------
    # Common spelling variations
    # --------------------------------------------------------

    u = u.replace("microliter", "µl")
    u = u.replace("microlitre", "µl")

    u = u.replace("milliliter", "ml")
    u = u.replace("millilitre", "ml")

    # --------------------------------------------------------
    # Cubic millimeter / cubic mm
    #
    # 1 µL = 1 cubic mm
    # --------------------------------------------------------

    cubic_variants = [
        "cumm",
        "cu.mm",
        "cu.mm.",
        "cu_mm",
        "cubicmm",
        "cubic-mm",
        "cubicmillimeter",
        "cubicmillimetre",
        "cubicmillimeters",
        "cubicmillimetres",
    ]

    for variant in cubic_variants:
        u = u.replace(variant, "µl")

    # --------------------------------------------------------
    # Hb conventions
    #
    # gm% is commonly used for g/dL.
    # --------------------------------------------------------

    u = u.replace("gm%", "g/dl")
    u = u.replace("gms%", "g/dl")
    u = u.replace("g%", "g/dl")

    # --------------------------------------------------------
    # Common lakh spellings
    # --------------------------------------------------------

    u = u.replace("lakhs", "lakh")
    u = u.replace("lac", "lakh")
    u = u.replace("lacs", "lakh")

    return u


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(value) -> Optional[float]:

    if value is None:
        return None

    try:
        result = float(value)

        if result != result:  # NaN
            return None

        return result

    except (TypeError, ValueError):
        return None


# ============================================================
# CONVERSION RESULT
# ============================================================

def conversion(
    value: Optional[float],
    unit: str,
    note: str,
    converted: bool,
    review: bool,
):
    return (
        value,
        unit,
        converted,
        note,
        review,
    )


# ============================================================
# UNIT CONVERSION ENGINE
# ============================================================

def convert_to_canonical(
    field: str,
    value: Optional[float],
    unit: Optional[str],
):
    """
    Convert report value into the canonical unit expected by
    the ML model.

    Returns:

        normalized_value
        normalized_unit
        conversion_applied
        conversion_note
        needs_review
    """

    value = safe_float(value)
    raw_unit = normalize_unit(unit)

    canonical = CANONICAL_UNITS[field]

    # ========================================================
    # MISSING VALUE
    # ========================================================

    if value is None:
        return conversion(
            None,
            canonical,
            "Missing value.",
            False,
            True,
        )

    # ========================================================
    # Hb
    # ========================================================

    if field == "Hb":

        # gm%, g%, g/dL
        if raw_unit in {
            "g/dl",
            "g/dl.",
        }:
            return conversion(
                value,
                "g/dL",
                "Converted/stabilized Hb unit to g/dL.",
                True,
                False,
            )

        # g/L -> g/dL
        if raw_unit == "g/l":
            return conversion(
                value / 10.0,
                "g/dL",
                "Converted g/L → g/dL.",
                True,
                False,
            )

        # Unit missing
        if not raw_unit and 4 <= value <= 25:
            return conversion(
                value,
                "g/dL",
                "Unit missing; interpreted as g/dL.",
                True,
                True,
            )

        return conversion(
            value,
            "g/dL",
            f"Unrecognized Hb unit: {unit}",
            False,
            True,
        )

    # ========================================================
    # RBC
    # ========================================================

    if field == "RBC":

        # million/cumm, million/µL, million/uL
        if raw_unit in {
            "million/µl",
            "million/ul",
            "million/μl",
            "million/uµl",
        }:
            return conversion(
                value,
                "million/µL",
                "Standardized RBC unit.",
                True,
                False,
            )

        # 10^12/L is numerically equivalent to million/µL
        if raw_unit in {
            "10^12/l",
            "x10^12/l",
            "10¹²/l",
            "x10¹²/l",
        }:
            return conversion(
                value,
                "million/µL",
                "Converted 10^12/L → million/µL.",
                True,
                False,
            )

        # Unit missing
        if not raw_unit and 1 <= value <= 10:
            return conversion(
                value,
                "million/µL",
                "Unit missing; interpreted as million/µL.",
                True,
                True,
            )

        return conversion(
            value,
            "million/µL",
            f"Unrecognized RBC unit: {unit}",
            False,
            True,
        )

    # ========================================================
    # WBC
    # ========================================================

    if field == "WBC":

        # Direct count
        if raw_unit in {
            "/µl",
            "µl",
            "cells/µl",
            "cell/µl",
            "perµl",
        }:
            return conversion(
                value,
                "/µL",
                "Standardized WBC count unit.",
                True,
                False,
            )

        # 10^3/µL
        if raw_unit in {
            "10^3/µl",
            "x10^3/µl",
            "10³/µl",
            "x10³/µl",
        }:
            return conversion(
                value * 1000,
                "/µL",
                "Converted ×10^3/µL → /µL.",
                True,
                False,
            )

        # 10^9/L = 10^3/µL
        if raw_unit in {
            "10^9/l",
            "x10^9/l",
            "10⁹/l",
            "x10⁹/l",
        }:
            return conversion(
                value * 1000,
                "/µL",
                "Converted ×10^9/L → /µL.",
                True,
                False,
            )

        # Unit missing
        if not raw_unit:

            if value >= 100:
                return conversion(
                    value,
                    "/µL",
                    "Unit missing; interpreted as /µL.",
                    True,
                    True,
                )

            if 0.5 <= value <= 50:
                return conversion(
                    value * 1000,
                    "/µL",
                    "Unit missing; interpreted as ×10^3/µL.",
                    True,
                    True,
                )

        return conversion(
            value,
            "/µL",
            f"Unrecognized WBC unit: {unit}",
            False,
            True,
        )

    # ========================================================
    # PLATELETS
    # ========================================================

    if field == "Platelets":

        # Direct count
        if raw_unit in {
            "/µl",
            "µl",
            "cells/µl",
            "cell/µl",
            "perµl",
        }:
            return conversion(
                value,
                "/µL",
                "Standardized platelet count unit.",
                True,
                False,
            )

        # 10^3/µL
        if raw_unit in {
            "10^3/µl",
            "x10^3/µl",
            "10³/µl",
            "x10³/µl",
        }:
            return conversion(
                value * 1000,
                "/µL",
                "Converted ×10^3/µL → /µL.",
                True,
                False,
            )

        # 10^9/L
        if raw_unit in {
            "10^9/l",
            "x10^9/l",
            "10⁹/l",
            "x10⁹/l",
        }:
            return conversion(
                value * 1000,
                "/µL",
                "Converted ×10^9/L → /µL.",
                True,
                False,
            )

        # lakh/µL
        #
        # 1 lakh = 100,000
        #
        # 1.52 lakh/µL
        #       ↓
        # 152,000 /µL
        if raw_unit in {
            "lakh/µl",
            "lakh/ul",
            "lakh/μl",
            "lakhperµl",
        }:
            return conversion(
                value * 100000,
                "/µL",
                "Converted lakh/µL → /µL.",
                True,
                False,
            )

        # Unit missing
        if not raw_unit:

            # Example: 152000
            if value >= 10000:
                return conversion(
                    value,
                    "/µL",
                    "Unit missing; interpreted as /µL.",
                    True,
                    True,
                )

            # Example: 152 = 152 x10^3/µL
            if 50 <= value <= 1000:
                return conversion(
                    value * 1000,
                    "/µL",
                    "Unit missing; interpreted as ×10^3/µL.",
                    True,
                    True,
                )

            # Example: 1.52 lakh/µL
            if 0.1 <= value <= 10:
                return conversion(
                    value * 100000,
                    "/µL",
                    "Unit missing; interpreted as lakh/µL.",
                    True,
                    True,
                )

        return conversion(
            value,
            "/µL",
            f"Unrecognized platelet unit: {unit}",
            False,
            True,
        )

    # ========================================================
    # DIFFERENTIALS
    # ========================================================

    if field in {
        "Neutrophils",
        "Lymphocytes",
        "Monocytes",
        "Eosinophils",
        "Basophils",
    }:

        if raw_unit in {
            "%",
            "percent",
            "percentage",
        }:
            return conversion(
                value,
                "%",
                "Standardized differential unit.",
                True,
                False,
            )

        # Missing unit
        if not raw_unit:

            if 0 <= value <= 1:
                return conversion(
                    value * 100,
                    "%",
                    "Unit missing; interpreted as fraction.",
                    True,
                    True,
                )

            if 1 < value <= 100:
                return conversion(
                    value,
                    "%",
                    "Unit missing; interpreted as percentage.",
                    True,
                    True,
                )

        return conversion(
            value,
            "%",
            f"Unrecognized differential unit: {unit}",
            False,
            True,
        )

    # ========================================================
    # MCV
    # ========================================================

    if field == "MCV":

        if raw_unit in {
            "fl",
            "femtoliter",
            "femtolitre",
            "um3",
            "µm3",
        }:
            return conversion(
                value,
                "fL",
                "Standardized MCV unit.",
                True,
                False,
            )

        if not raw_unit and 40 <= value <= 150:
            return conversion(
                value,
                "fL",
                "Unit missing; interpreted as fL.",
                True,
                True,
            )

        return conversion(
            value,
            "fL",
            f"Unrecognized MCV unit: {unit}",
            False,
            True,
        )

    # ========================================================
    # MCH
    # ========================================================

    if field == "MCH":

        if raw_unit in {
            "pg",
            "picogram",
            "picograms",
        }:
            return conversion(
                value,
                "pg",
                "Standardized MCH unit.",
                True,
                False,
            )

        if not raw_unit and 10 <= value <= 50:
            return conversion(
                value,
                "pg",
                "Unit missing; interpreted as pg.",
                True,
                True,
            )

        return conversion(
            value,
            "pg",
            f"Unrecognized MCH unit: {unit}",
            False,
            True,
        )

    # ========================================================
    # MCHC
    # ========================================================

    if field == "MCHC":

        if raw_unit in {
            "g/dl",
            "gdl",
        }:
            return conversion(
                value,
                "g/dL",
                "Standardized MCHC unit.",
                True,
                False,
            )

        if raw_unit == "g/l":
            return conversion(
                value / 10.0,
                "g/dL",
                "Converted g/L → g/dL.",
                True,
                False,
            )

        if not raw_unit and 20 <= value <= 45:
            return conversion(
                value,
                "g/dL",
                "Unit missing; interpreted as g/dL.",
                True,
                True,
            )

        return conversion(
            value,
            "g/dL",
            f"Unrecognized MCHC unit: {unit}",
            False,
            True,
        )

    # ========================================================
    # RDW
    # ========================================================

    if field == "RDW":

        if raw_unit in {
            "%",
            "percent",
            "percentage",
        }:
            return conversion(
                value,
                "%",
                "Standardized RDW unit.",
                True,
                False,
            )

        if not raw_unit:

            if 0 <= value <= 1:
                return conversion(
                    value * 100,
                    "%",
                    "Unit missing; interpreted as fraction.",
                    True,
                    True,
                )

            if 5 <= value <= 50:
                return conversion(
                    value,
                    "%",
                    "Unit missing; interpreted as percentage.",
                    True,
                    True,
                )

        return conversion(
            value,
            "%",
            f"Unrecognized RDW unit: {unit}",
            False,
            True,
        )

    # ========================================================
    # FALLBACK
    # ========================================================

    return conversion(
        value,
        CANONICAL_UNITS[field],
        "No conversion rule available.",
        False,
        True,
    )


# ============================================================
# PLAUSIBILITY CHECK
# ============================================================
#
# These are NOT clinical reference ranges.
# They are only sanity checks for parser/unit mistakes.
# ============================================================

PLAUSIBILITY_RANGES = {
    "Hb": (2, 25),
    "RBC": (1, 10),
    "WBC": (500, 100000),
    "Platelets": (5000, 2000000),
    "Neutrophils": (0, 100),
    "Lymphocytes": (0, 100),
    "Monocytes": (0, 100),
    "Eosinophils": (0, 100),
    "Basophils": (0, 100),
    "MCV": (40, 150),
    "MCH": (10, 50),
    "MCHC": (20, 50),
    "RDW": (5, 50),
}


def plausibility_check(
    field: str,
    value: Optional[float],
) -> Optional[str]:

    if value is None:
        return "Missing value."

    low, high = PLAUSIBILITY_RANGES[field]

    if value < low or value > high:
        return (
            f"{field} normalized value {value} is outside "
            f"the parser plausibility range {low}–{high}."
        )

    return None


# ============================================================
# NORMALIZE COMPLETE EXTRACTION
# ============================================================

def normalize_extraction(
    extracted: CBCExtraction,
) -> dict:

    extracted_data = extracted.model_dump()

    raw_values = {}
    normalized_values = {}
    normalized_units = {}

    conversions = []
    review_flags = []
    missing_fields = []

    for field in CBC_FIELDS:

        measurement = extracted_data[field]

        raw_value = safe_float(
            measurement.get("value")
        )

        raw_unit = measurement.get("unit")

        # ----------------------------------------------------
        # Preserve exactly what Gemini extracted
        # ----------------------------------------------------

        raw_values[field] = {
            "value": raw_value,
            "unit": raw_unit,
        }

        # ----------------------------------------------------
        # Missing
        # ----------------------------------------------------

        if raw_value is None:

            normalized_values[field] = None

            normalized_units[field] = (
                CANONICAL_UNITS[field]
            )

            missing_fields.append(field)

            review_flags.append({
                "type": "missing_value",
                "field": field,
                "severity": "attention",
                "message": (
                    f"{field} could not be confidently "
                    "extracted from the report."
                ),
            })

            continue

        # ----------------------------------------------------
        # Conversion
        # ----------------------------------------------------

        (
            normalized_value,
            normalized_unit,
            conversion_applied,
            conversion_note,
            needs_review,
        ) = convert_to_canonical(
            field,
            raw_value,
            raw_unit,
        )

        normalized_values[field] = normalized_value
        normalized_units[field] = normalized_unit

        # ----------------------------------------------------
        # Record conversion
        # ----------------------------------------------------

        if conversion_applied:

            conversions.append({
                "field": field,
                "from_value": raw_value,
                "from_unit": raw_unit,
                "to_value": normalized_value,
                "to_unit": normalized_unit,
                "note": conversion_note,
            })

        # ----------------------------------------------------
        # Review flag
        # ----------------------------------------------------

        if needs_review:

            review_flags.append({
                "type": "unit_or_interpretation_review",
                "field": field,
                "severity": "attention",
                "message": conversion_note,
                "raw_value": raw_value,
                "raw_unit": raw_unit,
                "normalized_value": normalized_value,
                "normalized_unit": normalized_unit,
            })

        # ----------------------------------------------------
        # Plausibility check
        # ----------------------------------------------------

        plausibility_error = plausibility_check(
            field,
            normalized_value,
        )

        if plausibility_error:

            review_flags.append({
                "type": "plausibility_warning",
                "field": field,
                "severity": "attention",
                "message": plausibility_error,
                "normalized_value": normalized_value,
                "normalized_unit": normalized_unit,
            })

    # --------------------------------------------------------
    # Manual review is ALWAYS required before model inference
    # --------------------------------------------------------

    requires_manual_review = True

    if missing_fields:
        review_reason = (
            "Some CBC values are missing and must be reviewed."
        )

    elif review_flags:
        review_reason = (
            "Some values or unit conversions require review "
            "before model assessment."
        )

    else:
        review_reason = (
            "Values were successfully normalized, but "
            "manual verification is still required before "
            "model assessment."
        )

    return {
        "raw_values": raw_values,
        "values": normalized_values,
        "units": normalized_units,
        "missing_fields": missing_fields,
        "conversions": conversions,
        "review_flags": review_flags,
        "requires_manual_review": requires_manual_review,
        "review_reason": review_reason,
    }


# ============================================================
# MAIN EXTRACTION FUNCTION
# ============================================================

def extract_cbc_from_report(
    file_path: str,
) -> dict:

    path = validate_file(file_path)

    print(
        f"\n📄 Report: {path.name}"
    )

    print(
        "📤 Uploading report to Gemini..."
    )

    uploaded_file = client.files.upload(
        file=str(path)
    )

    print(
        "✅ Report uploaded"
    )

    print(
        "🔎 Extracting CBC values and units..."
    )

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            EXTRACTION_PROMPT,
            uploaded_file,
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": CBCExtraction,
        },
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    # --------------------------------------------------------
    # Validate structured Gemini response
    # --------------------------------------------------------

    extracted = CBCExtraction.model_validate_json(
        response.text
    )

    # --------------------------------------------------------
    # Normalize units
    # --------------------------------------------------------

    normalized = normalize_extraction(
        extracted
    )

    return {
        "success": True,

        "source_file":
            path.name,

        "values":
            normalized["values"],

        "units":
            normalized["units"],

        "raw_values":
            normalized["raw_values"],

        "missing_fields":
            normalized["missing_fields"],

        "conversions":
            normalized["conversions"],

        "review_flags":
            normalized["review_flags"],

        "requires_manual_review":
            normalized["requires_manual_review"],

        "review_reason":
            normalized["review_reason"],

        "message":
            (
                "CBC values were extracted and normalized "
                "to the canonical units expected by the "
                "Aarogyam CBC model. Verify the extracted "
                "values before running the assessment."
            ),
    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

def main():

    print(
        "\n" + "=" * 75
    )

    print(
        "AAROGYAM AI — CBC VISUAL REPORT EXTRACTOR"
    )

    print(
        "=" * 75
    )

    if len(sys.argv) < 2:

        print("\nUsage:")

        print(
            'python backend/cbc_report_parser.py '
            '"./report.pdf"'
        )

        sys.exit(1)

    report_path = sys.argv[1]

    try:

        result = extract_cbc_from_report(
            report_path
        )

        # ----------------------------------------------------
        # Canonical values
        # ----------------------------------------------------

        print(
            "\n" + "-" * 75
        )

        print(
            "NORMALIZED CBC VALUES — MODEL INPUT UNITS"
        )

        print(
            "-" * 75
        )

        for field in CBC_FIELDS:

            value = result[
                "values"
            ][field]

            unit = result[
                "units"
            ][field]

            print(
                f"{field:<18}"
                f"{str(value):<14}"
                f"{unit}"
            )

        # ----------------------------------------------------
        # Conversions
        # ----------------------------------------------------

        print(
            "\nUNIT CONVERSIONS"
        )

        if result["conversions"]:

            for item in result["conversions"]:

                print(
                    f"  • {item['field']}: "
                    f"{item['from_value']} "
                    f"{item['from_unit']} "
                    f"→ "
                    f"{item['to_value']} "
                    f"{item['to_unit']}"
                )

        else:

            print(
                "  None"
            )

        # ----------------------------------------------------
        # Missing
        # ----------------------------------------------------

        print(
            "\nMISSING FIELDS"
        )

        if result["missing_fields"]:

            for field in result["missing_fields"]:
                print(
                    f"  ⚠ {field}"
                )

        else:

            print(
                "  None"
            )

        # ----------------------------------------------------
        # Review flags
        # ----------------------------------------------------

        print(
            "\nREVIEW FLAGS"
        )

        if result["review_flags"]:

            for flag in result["review_flags"]:

                print(
                    f"  ⚠ {flag['field']}: "
                    f"{flag['message']}"
                )

        else:

            print(
                "  None"
            )

        # ----------------------------------------------------
        # Review status
        # ----------------------------------------------------

        print(
            "\nMANUAL REVIEW:"
        )

        print(
            "  REQUIRED"
            if result["requires_manual_review"]
            else
            "  Not required"
        )

        # ----------------------------------------------------
        # Full result
        # ----------------------------------------------------

        print(
            "\nFULL JSON"
        )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

    except Exception as error:

        print(
            "\n❌ EXTRACTION FAILED"
        )

        print(
            str(error)
        )

        sys.exit(1)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
