# ============================================================
# AAROGYAM AI
# CBC TEXT REPORT EXTRACTOR + UNIT NORMALIZER
# ============================================================
#
# PURPOSE:
#   1. Read a text-based CBC PDF.
#   2. Extract the patient's CBC RESULT values.
#   3. Extract the units shown on the report.
#   4. Normalize different laboratory units into the
#      exact units expected by the CBC ML model.
#   5. Flag ambiguous/unrecognized values for manual review.
#
# IMPORTANT:
#   This file ONLY extracts and normalizes laboratory values.
#
#   It does NOT:
#   - diagnose the patient
#   - predict diseases
#   - calculate severity
#   - interpret abnormal results
#   - run XGBoost/LightGBM
#
# FLOW:
#
#   PDF
#    ↓
#   pdfplumber
#    ↓
#   CBC text extraction
#    ↓
#   CBC field matching
#    ↓
#   Unit normalization
#    ↓
#   Doctor review
#    ↓
#   /api/cbc/assess
#
# NOTE:
#   This version intentionally avoids Gemini for normal
#   text-based PDF CBC reports.
#
# ============================================================

import json
import re
import sys
from pathlib import Path
from typing import Optional

import pdfplumber


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
# FIELD ALIASES
# ============================================================
#
# These are the report labels that map to your ML fields.
# ============================================================

FIELD_ALIASES = {
    "Hb": [
        "haemoglobin",
        "hemoglobin",
        "hgb",
        "hb",
    ],

    "RBC": [
        "rbc count",
        "rbc",
        "red blood cell count",
        "red blood cells",
        "erythrocytes",
    ],

    "WBC": [
        "total leucocyte count",
        "total leukocyte count",
        "tlc",
        "wbc count",
        "wbc",
        "white blood cell count",
        "white blood cells",
        "leukocytes",
    ],

    "Platelets": [
        "platelet count",
        "platelets",
        "platelet",
        "plt",
    ],

    "Neutrophils": [
        "neutrophils",
        "neutrophil",
        "neutrophils %",
        "neutrophil %",
        "neut",
        "neu",
    ],

    "Lymphocytes": [
        "lymphocytes",
        "lymphocyte",
        "lymphocytes %",
        "lymphocyte %",
        "lymph",
        "lym",
    ],

    "Monocytes": [
        "monocytes",
        "monocyte",
        "monocytes %",
        "monocyte %",
        "mono",
        "mon",
    ],

    "Eosinophils": [
        "eosinophils",
        "eosinophil",
        "eosinophils %",
        "eosinophil %",
        "eos",
    ],

    "Basophils": [
        "basophils",
        "basophil",
        "basophils %",
        "basophil %",
        "baso",
        "bas",
    ],

    "MCV": [
        "mcv",
        "mean corpuscular volume",
    ],

    "MCH": [
        "mch",
        "mean corpuscular hemoglobin",
        "mean corpuscular haemoglobin",
    ],

    "MCHC": [
        "mchc",
        "mean corpuscular hemoglobin concentration",
        "mean corpuscular haemoglobin concentration",
    ],

    "RDW": [
        "rdw-cv",
        "rdw cv",
        "rdw",
        "red cell distribution width",
    ],
}


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
}


# ============================================================
# FILE VALIDATION
# ============================================================

def validate_file(file_path: str) -> Path:
    """
    Validate the uploaded file.

    Normal CBC extraction in this version is supported for
    text-based PDF reports.

    Images are accepted by the API layer but require OCR,
    which is intentionally not performed here.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    extension = path.suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. "
            "Use PDF, PNG, JPG, or JPEG."
        )

    return path


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(value) -> Optional[float]:
    """
    Convert a value safely to float.
    """

    if value is None:
        return None

    try:
        text = str(value).strip()

        # Remove thousands separators.
        text = text.replace(",", "")

        result = float(text)

        # NaN check.
        if result != result:
            return None

        return result

    except (TypeError, ValueError):
        return None


# ============================================================
# UNIT TEXT NORMALIZATION
# ============================================================

def normalize_unit(unit: Optional[str]) -> str:
    """
    Normalize visually different representations of units.

    Examples:

        gm%            -> g/dl
        Million/cumm   -> million/µl
        /cumm          -> /µl
        lakh/cumm      -> lakh/µl
        x10^3/µL       -> x10^3/µl
    """

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
    # Normalize spaces
    # --------------------------------------------------------

    u = " ".join(u.split())
    u = u.replace(" ", "")

    # --------------------------------------------------------
    # Common spelling variations
    # --------------------------------------------------------

    u = u.replace("microliter", "µl")
    u = u.replace("microlitre", "µl")

    # --------------------------------------------------------
    # Cubic millimeter / cubic mm
    #
    # 1 µL = 1 cubic mm
    # --------------------------------------------------------

    cubic_variants = [
        "cumm",
        "cu.mm.",
        "cu.mm",
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
    u = u.replace("lacs", "lakh")
    u = u.replace("lac", "lakh")

    return u


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

        if raw_unit in {
            "g/dl",
            "g/dl.",
        }:

            return conversion(
                value,
                "g/dL",
                "Standardized Hb unit to g/dL.",
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

        if raw_unit in {
            "million/µl",
            "million/ul",
        }:

            return conversion(
                value,
                "million/µL",
                "Standardized RBC unit.",
                True,
                False,
            )

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

        if raw_unit in {
            "lakh/µl",
            "lakh/ul",
            "lakhperµl",
        }:

            return conversion(
                value * 100000,
                "/µL",
                "Converted lakh/µL → /µL.",
                True,
                False,
            )

        if not raw_unit:

            if value >= 10000:

                return conversion(
                    value,
                    "/µL",
                    "Unit missing; interpreted as /µL.",
                    True,
                    True,
                )

            if 50 <= value <= 1000:

                return conversion(
                    value * 1000,
                    "/µL",
                    "Unit missing; interpreted as ×10^3/µL.",
                    True,
                    True,
                )

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
        canonical,
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
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(file_path: str) -> str:
    """
    Extract text from all PDF pages using pdfplumber.
    """

    path = validate_file(file_path)

    if path.suffix.lower() != ".pdf":

        raise ValueError(
            "This parser currently supports text-based PDF "
            "CBC reports. Image OCR is not enabled."
        )

    pages_text = []

    try:

        with pdfplumber.open(path) as pdf:

            if not pdf.pages:

                raise ValueError(
                    "The PDF contains no pages."
                )

            for page_number, page in enumerate(
                pdf.pages,
                start=1,
            ):

                text = page.extract_text()

                if text:

                    pages_text.append(
                        text
                    )

    except Exception as error:

        raise RuntimeError(
            f"Unable to read PDF: {error}"
        ) from error

    full_text = "\n".join(
        pages_text
    )

    if not full_text.strip():

        raise ValueError(
            "No readable text was found in this PDF. "
            "This may be a scanned/image-only report."
        )

    return full_text


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_line(line: str) -> str:
    """
    Normalize whitespace while keeping the actual content.
    """

    line = line.replace("\xa0", " ")

    line = re.sub(
        r"\s+",
        " ",
        line,
    )

    return line.strip()


# ============================================================
# FIELD LABEL DETECTION
# ============================================================

def find_field_for_line(line: str) -> Optional[str]:
    """
    Determine whether a line represents one of the CBC fields.

    Matching is done against the beginning of the line so that
    things such as 'Absolute Neutrophils' do not accidentally
    become 'Neutrophils'.
    """

    cleaned = normalize_line(line)

    lower = cleaned.lower()

    # Longest aliases first to prevent:
    #
    #   "rbc count"
    #
    # from being interpreted as just "rbc".

    candidates = []

    for field, aliases in FIELD_ALIASES.items():

        for alias in aliases:

            candidates.append(
                (
                    field,
                    alias,
                )
            )

    candidates.sort(
        key=lambda item: len(item[1]),
        reverse=True,
    )

    for field, alias in candidates:

        pattern = (
            r"^"
            + re.escape(alias)
            + r"(?=\s|:|$)"
        )

        if re.search(
            pattern,
            lower,
        ):

            return field

    return None


# ============================================================
# UNIT DETECTION
# ============================================================

UNIT_PATTERNS = [
    # More specific units first.
    r"million\s*/\s*cumm",
    r"million\s*/\s*ul",
    r"million\s*/\s*µl",
    r"lakh\s*/\s*cumm",
    r"lakh\s*/\s*ul",
    r"lakh\s*/\s*µl",

    r"x\s*10\^3\s*/\s*µl",
    r"x\s*10\^3\s*/\s*ul",
    r"10\^3\s*/\s*µl",
    r"10\^3\s*/\s*ul",

    r"x\s*10\^9\s*/\s*l",
    r"10\^9\s*/\s*l",

    r"gm\s*%",
    r"gms\s*%",
    r"g\s*%",

    r"g\s*/\s*dl",
    r"g\s*/\s*l",

    r"/\s*cumm",
    r"/\s*ul",
    r"/\s*µl",

    r"cells\s*/\s*µl",
    r"cells\s*/\s*ul",

    r"cell\s*/\s*µl",
    r"cell\s*/\s*ul",

    r"per\s*µl",
    r"per\s*ul",

    r"fL",
    r"fl",

    r"pg",

    r"%",

    r"percent",
    r"percentage",
]


def extract_unit_from_line(line: str) -> Optional[str]:
    """
    Find the laboratory unit near the end of the line.

    We deliberately prefer the last unit because reference
    ranges can also contain '%' or other symbols.
    """

    cleaned = normalize_line(line)

    matches = []

    for pattern in UNIT_PATTERNS:

        for match in re.finditer(
            pattern,
            cleaned,
            flags=re.IGNORECASE,
        ):

            matches.append(
                match
            )

    if not matches:
        return None

    # Use the right-most match.
    match = max(
        matches,
        key=lambda item: item.end(),
    )

    # Only accept a unit that occurs close to the end.
    trailing = cleaned[
        match.end():
    ].strip()

    # A CBC line normally has the unit as the final token.
    if trailing:
        return None

    return match.group(0)


# ============================================================
# NUMERIC VALUE EXTRACTION
# ============================================================

def extract_first_numeric_value(
    text: str,
) -> Optional[float]:
    """
    Extract the first numeric value from the portion of a
    CBC line after its field label.

    This is intentional.

    Example:

        Haemoglobin 13.0 11.5 - 18.0 gm%

    The first number is the patient result:

        13.0

    while the later numbers are the biological reference
    range and must be ignored.
    """

    number_pattern = (
        r"(?<![A-Za-z])"
        r"-?"
        r"\d+(?:[.,]\d+)?"
        r"(?![A-Za-z])"
    )

    match = re.search(
        number_pattern,
        text,
    )

    if not match:
        return None

    return safe_float(
        match.group(0)
    )


# ============================================================
# REMOVE FIELD LABEL
# ============================================================

def remove_field_label(
    line: str,
    field: str,
) -> str:
    """
    Remove the recognized field label from the beginning
    of a CBC line.
    """

    cleaned = normalize_line(line)

    lower = cleaned.lower()

    aliases = sorted(
        FIELD_ALIASES[field],
        key=len,
        reverse=True,
    )

    for alias in aliases:

        pattern = (
            r"^"
            + re.escape(alias)
            + r"(?=\s|:|$)"
        )

        match = re.match(
            pattern,
            lower,
        )

        if match:

            return cleaned[
                match.end():
            ].strip()

    return cleaned


# ============================================================
# PARSE CBC TEXT
# ============================================================

def parse_cbc_text(
    text: str,
) -> dict:
    """
    Parse CBC values from extracted PDF text.

    This parser intentionally ignores:
        - patient metadata
        - absolute leukocyte counts
        - reference ranges
        - Widal/serology
        - administrative text
    """

    extracted = {}

    lines = text.splitlines()

    for raw_line in lines:

        line = normalize_line(
            raw_line
        )

        if not line:
            continue

        field = find_field_for_line(
            line
        )

        if field is None:
            continue

        # ----------------------------------------------------
        # Ignore duplicates once we have a confident result.
        # ----------------------------------------------------

        if field in extracted:
            continue

        remainder = remove_field_label(
            line,
            field,
        )

        if not remainder:
            continue

        unit = extract_unit_from_line(
            remainder
        )

        value = extract_first_numeric_value(
            remainder
        )

        if value is None:
            continue

        extracted[field] = {
            "value": value,
            "unit": unit,
            "source_line": line,
        }

    return extracted


# ============================================================
# NORMALIZE COMPLETE EXTRACTION
# ============================================================

def normalize_extraction(
    extracted: dict,
) -> dict:
    """
    Convert extracted raw values into the canonical units
    expected by the ML model.
    """

    raw_values = {}

    normalized_values = {}

    normalized_units = {}

    conversions = []

    review_flags = []

    missing_fields = []

    # --------------------------------------------------------
    # Process every required CBC field.
    # --------------------------------------------------------

    for field in CBC_FIELDS:

        measurement = extracted.get(
            field
        )

        # ----------------------------------------------------
        # Missing field
        # ----------------------------------------------------

        if measurement is None:

            raw_values[field] = {
                "value": None,
                "unit": None,
            }

            normalized_values[field] = None

            normalized_units[field] = (
                CANONICAL_UNITS[field]
            )

            missing_fields.append(
                field
            )

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

        raw_value = safe_float(
            measurement.get("value")
        )

        raw_unit = measurement.get(
            "unit"
        )

        # ----------------------------------------------------
        # Preserve raw value/unit
        # ----------------------------------------------------

        raw_values[field] = {
            "value": raw_value,
            "unit": raw_unit,
        }

        # ----------------------------------------------------
        # Convert
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

        normalized_values[field] = (
            normalized_value
        )

        normalized_units[field] = (
            normalized_unit
        )

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
                "normalized_value":
                    normalized_value,
                "normalized_unit":
                    normalized_unit,
            })

        # ----------------------------------------------------
        # Plausibility check
        # ----------------------------------------------------

        plausibility_error = (
            plausibility_check(
                field,
                normalized_value,
            )
        )

        if plausibility_error:

            review_flags.append({
                "type": "plausibility_warning",
                "field": field,
                "severity": "attention",
                "message": plausibility_error,
                "normalized_value":
                    normalized_value,
                "normalized_unit":
                    normalized_unit,
            })

    # --------------------------------------------------------
    # Manual review is always required before model inference.
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
        "requires_manual_review":
            requires_manual_review,
        "review_reason": review_reason,
    }


# ============================================================
# MAIN EXTRACTION FUNCTION
# ============================================================

def extract_cbc_from_report(
    file_path: str,
) -> dict:
    """
    Main CBC extraction entry point used by backend/main.py.
    """

    path = validate_file(
        file_path
    )

    print(
        f"\n📄 Report: {path.name}"
    )

    print(
        "📖 Reading CBC PDF locally with pdfplumber..."
    )

    text = extract_pdf_text(
        str(path)
    )

    print(
        "✅ PDF text extracted"
    )

    print(
        "🔎 Searching for CBC values..."
    )

    parsed = parse_cbc_text(
        text
    )

    # --------------------------------------------------------
    # Diagnostic output
    # --------------------------------------------------------

    print(
        f"✅ Found {len(parsed)} CBC fields"
    )

    for field in CBC_FIELDS:

        if field in parsed:

            print(
                f"   {field}: "
                f"{parsed[field]['value']} "
                f"{parsed[field]['unit']}"
            )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized = normalize_extraction(
        parsed
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

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
                "CBC values were extracted from the PDF "
                "and normalized to the canonical units "
                "expected by the Aarogyam CBC model. "
                "Verify the extracted values before "
                "running the assessment."
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
        "AAROGYAM AI — CBC PDF TEXT EXTRACTOR"
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
        # Unit conversions
        # ----------------------------------------------------

        print(
            "\nUNIT CONVERSIONS"
        )

        if result["conversions"]:

            for item in result[
                "conversions"
            ]:

                print(
                    f"  • {item['field']}: "
                    f"{item['from_value']} "
                    f"{item['from_unit']} "
                    f"→ "
                    f"{item['to_value']} "
                    f"{item['to_unit']} "
                    f"({item['note']})"
                )

        else:

            print(
                "  None"
            )

        # ----------------------------------------------------
        # Missing fields
        # ----------------------------------------------------

        print(
            "\nMISSING FIELDS"
        )

        if result["missing_fields"]:

            for field in result[
                "missing_fields"
            ]:

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

            for flag in result[
                "review_flags"
            ]:

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
            if result[
                "requires_manual_review"
            ]
            else
            "  Not required"
        )

        # ----------------------------------------------------
        # Full JSON
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