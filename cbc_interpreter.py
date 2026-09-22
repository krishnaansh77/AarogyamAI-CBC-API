# ============================================================
# AAROGYAM AI
# CBC INTERPRETATION / POST-PROCESSING LAYER
# ============================================================
#
# PURPOSE:
#   Convert raw CBC measurements into structured laboratory
#   interpretation for the frontend/API.
#
# THIS FILE DOES NOT:
#   - predict disease
#   - change ML probabilities
#   - diagnose dengue/malaria
#   - use SHAP to decide whether a value is "normal"
#
# THIS FILE DOES:
#   - classify CBC measurements as low/high/normal
#   - provide reference-range information
#   - create readable laboratory messages
#   - summarize RBC/WBC/platelet lineages
#   - create conservative attention flags
#   - create generic follow-up text
#
# IMPORTANT:
#   Reference ranges below are PROTOTYPE ranges.
#   They must be replaced with validated laboratory-specific
#   intervals before any clinical use.
# ============================================================

from typing import Dict, List, Optional, Any
import json


# ============================================================
# PROTOTYPE REFERENCE RANGES
# ============================================================
#
# These are deliberately stored in one place so that they can
# later be replaced by:
#
#   - laboratory-specific ranges
#   - age-specific ranges
#   - sex-specific ranges
#   - institution-specific configuration
#
# The current frontend/model does not include Sex.
# Therefore this version uses prototype general ranges.
#
# Units:
#   Hb          g/dL
#   RBC         million/µL
#   WBC         /µL
#   Platelets   /µL
#   Differential %
#   MCV         fL
#   MCH         pg
#   MCHC        g/dL
#   RDW         %
# ============================================================

REFERENCE_RANGES = {

    "Hb": {
        "unit": "g/dL",
        "low": 12.0,
        "high": 17.5,
        "critical_low": 7.0,
        "critical_high": None
    },

    "RBC": {
        "unit": "million/µL",
        "low": 4.0,
        "high": 6.0,
        "critical_low": None,
        "critical_high": None
    },

    "WBC": {
        "unit": "/µL",
        "low": 4000,
        "high": 11000,
        "critical_low": 2000,
        "critical_high": 30000
    },

    "Platelets": {
        "unit": "/µL",
        "low": 150000,
        "high": 450000,
        "critical_low": 50000,
        "critical_high": 1000000
    },

    "Neutrophils": {
        "unit": "%",
        "low": 40,
        "high": 70,
        "critical_low": None,
        "critical_high": None
    },

    "Lymphocytes": {
        "unit": "%",
        "low": 20,
        "high": 40,
        "critical_low": None,
        "critical_high": None
    },

    "Monocytes": {
        "unit": "%",
        "low": 2,
        "high": 10,
        "critical_low": None,
        "critical_high": None
    },

    "Eosinophils": {
        "unit": "%",
        "low": 0,
        "high": 6,
        "critical_low": None,
        "critical_high": None
    },

    "Basophils": {
        "unit": "%",
        "low": 0,
        "high": 2,
        "critical_low": None,
        "critical_high": None
    },

    "MCV": {
        "unit": "fL",
        "low": 80,
        "high": 100,
        "critical_low": None,
        "critical_high": None
    },

    "MCH": {
        "unit": "pg",
        "low": 27,
        "high": 33,
        "critical_low": None,
        "critical_high": None
    },

    "MCHC": {
        "unit": "g/dL",
        "low": 32,
        "high": 36,
        "critical_low": None,
        "critical_high": None
    },

    "RDW": {
        "unit": "%",
        "low": 11.5,
        "high": 14.5,
        "critical_low": None,
        "critical_high": None
    }
}


# ============================================================
# DISPLAY INFORMATION
# ============================================================

PARAMETER_LABELS = {

    "Hb": "Hemoglobin",

    "RBC": "RBC",

    "WBC": "WBC",

    "Platelets": "Platelets",

    "Neutrophils": "Neutrophils",

    "Lymphocytes": "Lymphocytes",

    "Monocytes": "Monocytes",

    "Eosinophils": "Eosinophils",

    "Basophils": "Basophils",

    "MCV": "MCV",

    "MCH": "MCH",

    "MCHC": "MCHC",

    "RDW": "RDW"
}


# ============================================================
# PARAMETER GROUPS
# ============================================================

GROUPS = {

    "rbc": [
        "Hb",
        "RBC",
        "MCV",
        "MCH",
        "MCHC",
        "RDW"
    ],

    "wbc": [
        "WBC",
        "Neutrophils",
        "Lymphocytes",
        "Monocytes",
        "Eosinophils",
        "Basophils"
    ],

    "platelet": [
        "Platelets"
    ]
}


GROUP_LABELS = {

    "rbc":
        "Red Blood Cell Parameters",

    "wbc":
        "White Blood Cell Parameters",

    "platelet":
        "Platelet Parameters"
}


# ============================================================
# SAFE NUMBER CONVERSION
# ============================================================

def safe_float(value: Any) -> Optional[float]:

    if value is None:
        return None

    try:
        value = float(value)

        if value != value:   # NaN
            return None

        return value

    except (TypeError, ValueError):
        return None


# ============================================================
# FORMAT VALUE
# ============================================================

def format_value(
    value: float,
    parameter: str
) -> str:

    if parameter in [
        "WBC",
        "Platelets"
    ]:

        return f"{value:,.0f}"

    if parameter == "RBC":

        return f"{value:.2f}"

    return f"{value:.1f}"


# ============================================================
# DETERMINE STATUS
# ============================================================

def determine_status(
    parameter: str,
    value: Optional[float]
) -> str:

    if value is None:
        return "missing"

    if parameter not in REFERENCE_RANGES:
        return "unknown"

    reference = REFERENCE_RANGES[
        parameter
    ]

    low = reference["low"]

    high = reference["high"]

    critical_low = reference[
        "critical_low"
    ]

    critical_high = reference[
        "critical_high"
    ]

    # Critical conditions first
    if (
        critical_low is not None
        and value < critical_low
    ):
        return "critical"

    if (
        critical_high is not None
        and value > critical_high
    ):
        return "critical"

    if value < low:
        return "low"

    if value > high:
        return "high"

    return "normal"


# ============================================================
# STATUS MESSAGE
# ============================================================

def create_status_message(
    parameter: str,
    value: Optional[float],
    status: str
) -> str:

    label = PARAMETER_LABELS.get(
        parameter,
        parameter
    )

    if status == "missing":

        return (
            f"{label} value is missing."
        )

    if status == "low":

        return (
            f"{label} is below the "
            f"configured reference range."
        )

    if status == "high":

        return (
            f"{label} is above the "
            f"configured reference range."
        )

    if status == "critical":

        return (
            f"{label} is outside the "
            f"configured critical threshold."
        )

    if status == "normal":

        return (
            f"{label} is within the "
            f"configured reference range."
        )

    return (
        f"{label} could not be interpreted."
    )


# ============================================================
# REFERENCE STRING
# ============================================================

def reference_range_string(
    parameter: str
) -> str:

    reference = REFERENCE_RANGES[
        parameter
    ]

    low = reference["low"]

    high = reference["high"]

    return (
        f"{low:g}–{high:g}"
    )


# ============================================================
# INTERPRET ONE PARAMETER
# ============================================================

def interpret_parameter(
    parameter: str,
    value: Any
) -> Dict[str, Any]:

    numeric_value = safe_float(
        value
    )

    status = determine_status(
        parameter,
        numeric_value
    )

    reference = REFERENCE_RANGES.get(
        parameter,
        {}
    )

    return {

        "parameter":
            PARAMETER_LABELS.get(
                parameter,
                parameter
            ),

        "key":
            parameter,

        "value":
            numeric_value,

        "value_display":
            (
                format_value(
                    numeric_value,
                    parameter
                )
                if numeric_value is not None
                else None
            ),

        "unit":
            reference.get(
                "unit"
            ),

        "status":
            status,

        "reference_range":
            (
                reference_range_string(
                    parameter
                )
                if parameter in REFERENCE_RANGES
                else None
            ),

        "message":
            create_status_message(
                parameter,
                numeric_value,
                status
            )
    }


# ============================================================
# INTERPRET COMPLETE CBC
# ============================================================

def interpret_cbc(
    patient: Dict[str, Any]
) -> Dict[str, Any]:

    results = []

    # --------------------------------------------------------
    # Interpret every configured parameter
    # --------------------------------------------------------

    for parameter in REFERENCE_RANGES:

        value = patient.get(
            parameter
        )

        result = interpret_parameter(
            parameter,
            value
        )

        results.append(result)


    # --------------------------------------------------------
    # Group results
    # --------------------------------------------------------

    grouped_results = {

        "rbc": [
            result
            for result in results
            if result["key"] in GROUPS["rbc"]
        ],

        "wbc": [
            result
            for result in results
            if result["key"] in GROUPS["wbc"]
        ],

        "platelet": [
            result
            for result in results
            if result["key"] in GROUPS["platelet"]
        ]
    }


    # --------------------------------------------------------
    # Lineage overview
    # --------------------------------------------------------

    lineage_overview = create_lineage_overview(
        grouped_results
    )


    # --------------------------------------------------------
    # Abnormal values
    # --------------------------------------------------------

    abnormal_results = [
        result
        for result in results
        if result["status"] in [
            "low",
            "high",
            "critical"
        ]
    ]


    # --------------------------------------------------------
    # Critical values
    # --------------------------------------------------------

    critical_results = [
        result
        for result in results
        if result["status"] == "critical"
    ]


    # --------------------------------------------------------
    # Conservative severity flag
    # --------------------------------------------------------

    severity = determine_severity(
        abnormal_results,
        critical_results
    )


    # --------------------------------------------------------
    # Clinical flag
    # --------------------------------------------------------

    clinical_flag = create_clinical_flag(
        severity,
        critical_results,
        abnormal_results
    )


    # --------------------------------------------------------
    # Follow-up
    # --------------------------------------------------------

    follow_up = create_follow_up(
        abnormal_results,
        critical_results
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = create_cbc_summary(
        abnormal_results
    )


    return {

        "cbc_results":
            results,

        "grouped_results":
            grouped_results,

        "lineage_overview":
            lineage_overview,

        "abnormal_results":
            abnormal_results,

        "severity":
            severity,

        "clinical_flag_message":
            clinical_flag,

        "follow_up":
            follow_up,

        "pattern_summary":
            summary,

        "reference_range_source":
            "Prototype configurable ranges — "
            "replace with validated laboratory-specific "
            "ranges before clinical use."
    }


# ============================================================
# LINEAGE OVERVIEW
# ============================================================

def create_lineage_overview(
    grouped_results: Dict[str, List[Dict]]
) -> List[Dict[str, str]]:

    overview = []

    # --------------------------------------------------------
    # RBC
    # --------------------------------------------------------

    rbc_statuses = [
        item["status"]
        for item in grouped_results["rbc"]
    ]

    overview.append({
        "lineage":
            "RBC Status",

        "status":
            summarize_lineage(
                rbc_statuses
            )
    })


    # --------------------------------------------------------
    # WBC
    # --------------------------------------------------------

    wbc_statuses = [
        item["status"]
        for item in grouped_results["wbc"]
    ]

    overview.append({
        "lineage":
            "WBC Status",

        "status":
            summarize_lineage(
                wbc_statuses
            )
    })


    # --------------------------------------------------------
    # Platelets
    # --------------------------------------------------------

    platelet_statuses = [
        item["status"]
        for item in grouped_results["platelet"]
    ]

    overview.append({
        "lineage":
            "Platelet Status",

        "status":
            summarize_lineage(
                platelet_statuses
            )
    })

    return overview


# ============================================================
# SUMMARIZE LINEAGE
# ============================================================

def summarize_lineage(
    statuses: List[str]
) -> str:

    if "critical" in statuses:
        return "critical"

    if "low" in statuses and "high" in statuses:
        return "mixed abnormal"

    if "low" in statuses:
        return "low"

    if "high" in statuses:
        return "high"

    if all(
        status == "normal"
        for status in statuses
    ):
        return "normal"

    if "missing" in statuses:
        return "incomplete"

    return "mixed"


# ============================================================
# SEVERITY
# ============================================================

def determine_severity(
    abnormal_results: List[Dict],
    critical_results: List[Dict]
) -> str:

    """
    IMPORTANT:

    This is a PROTOTYPE alerting layer.

    It is NOT a validated clinical severity score.

    Current logic:
        critical finding      -> urgent flag
        3+ abnormal findings  -> attention
        1-2 abnormal findings -> attention
        no abnormality        -> routine

    In other words, this intentionally does NOT claim that
    an ML probability or an individual CBC result determines
    medical urgency.
    """

    if len(critical_results) > 0:
        return "urgent"

    if len(abnormal_results) > 0:
        return "attention"

    return "routine"


# ============================================================
# CLINICAL FLAG MESSAGE
# ============================================================

def create_clinical_flag(
    severity: str,
    critical_results: List[Dict],
    abnormal_results: List[Dict]
) -> str:

    if severity == "urgent":

        parameters = ", ".join(
            result["parameter"]
            for result in critical_results
        )

        return (
            "One or more CBC values fall outside "
            "the configured prototype critical "
            f"thresholds ({parameters}). "
            "Prompt clinical review is recommended."
        )

    if severity == "attention":

        return (
            "One or more CBC values are outside "
            "the configured prototype reference "
            "ranges and may warrant clinical review."
        )

    return (
        "No CBC value is outside the configured "
        "prototype reference ranges."
    )


# ============================================================
# FOLLOW-UP
# ============================================================

def create_follow_up(
    abnormal_results: List[Dict],
    critical_results: List[Dict]
) -> List[str]:

    recommendations = []

    # --------------------------------------------------------
    # Critical
    # --------------------------------------------------------

    if critical_results:

        recommendations.append(
            "Prompt clinical review is recommended "
            "for values outside the configured "
            "critical thresholds."
        )

    # --------------------------------------------------------
    # Any abnormality
    # --------------------------------------------------------

    if abnormal_results:

        recommendations.append(
            "Correlate CBC findings with the patient's "
            "clinical presentation and applicable "
            "laboratory reference intervals."
        )

        recommendations.append(
            "Consider repeat CBC or additional "
            "evaluation when clinically indicated."
        )

    # --------------------------------------------------------
    # Normal
    # --------------------------------------------------------

    if not abnormal_results:

        recommendations.append(
            "No CBC-specific follow-up is generated "
            "by this prototype interpreter."
        )

    return recommendations


# ============================================================
# CBC SUMMARY
# ============================================================

def create_cbc_summary(
    abnormal_results: List[Dict]
) -> str:

    if not abnormal_results:

        return (
            "The measured CBC parameters are within "
            "the configured prototype reference ranges."
        )

    low_values = [
        result["parameter"]
        for result in abnormal_results
        if result["status"] == "low"
    ]

    high_values = [
        result["parameter"]
        for result in abnormal_results
        if result["status"] == "high"
    ]

    critical_values = [
        result["parameter"]
        for result in abnormal_results
        if result["status"] == "critical"
    ]


    parts = []


    if low_values:

        parts.append(
            "Below range: "
            + ", ".join(low_values)
        )


    if high_values:

        parts.append(
            "Above range: "
            + ", ".join(high_values)
        )


    if critical_values:

        parts.append(
            "Critical threshold flag: "
            + ", ".join(critical_values)
        )


    return " | ".join(parts)


# ============================================================
# FRONTEND-FRIENDLY SUPPORTING EVIDENCE
# ============================================================

def create_supporting_evidence(
    interpretation: Dict[str, Any]
) -> List[Dict[str, Any]]:

    strong = []
    moderate = []

    for result in interpretation[
        "abnormal_results"
    ]:

        if result["status"] == "critical":

            strong.append(
                result["message"]
            )

        elif result["key"] in [
            "Hb",
            "WBC",
            "Platelets",
            "RBC"
        ]:

            strong.append(
                result["message"]
            )

        else:

            moderate.append(
                result["message"]
            )


    return [

        {
            "strength": "strong",
            "items": strong
        },

        {
            "strength": "moderate",
            "items": moderate
        }
    ]


# ============================================================
# COMPLETE INTERPRETATION
# ============================================================

def build_interpretation(
    patient: Dict[str, Any]
) -> Dict[str, Any]:

    interpretation = interpret_cbc(
        patient
    )

    interpretation[
        "supporting_evidence"
    ] = create_supporting_evidence(
        interpretation
    )

    return interpretation


# ============================================================
# TEST PATIENT
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("AAROGYAM AI — CBC INTERPRETER TEST")
    print("=" * 70)


    test_patient = {

        "Age": 24,
        "Height": 157,
        "Weight": 57,
        "BMI": 23.1,

        "Hb": 12.8,
        "RBC": 4.04,

        "WBC": 3800,

        "Platelets": 92000,

        "Neutrophils": 38,
        "Lymphocytes": 52,
        "Monocytes": 7,
        "Eosinophils": 2,
        "Basophils": 1,

        "MCV": 93.5,
        "MCH": 31.6,
        "MCHC": 33.8,
        "RDW": 12.5
    }


    result = build_interpretation(
        test_patient
    )


    # --------------------------------------------------------
    # Print individual results
    # --------------------------------------------------------

    print("\nCBC RESULTS\n")

    for item in result[
        "cbc_results"
    ]:

        print(
            f"{item['parameter']:<18}"
            f"{str(item['value_display']):<12}"
            f"{str(item['unit']):<12}"
            f"{item['status']:<10}"
        )


    # --------------------------------------------------------
    # Lineage
    # --------------------------------------------------------

    print(
        "\nLINEAGE OVERVIEW\n"
    )

    for item in result[
        "lineage_overview"
    ]:

        print(
            f"{item['lineage']:<20}"
            f"{item['status']}"
        )


    # --------------------------------------------------------
    # Abnormal findings
    # --------------------------------------------------------

    print(
        "\nABNORMAL FINDINGS\n"
    )

    if result[
        "abnormal_results"
    ]:

        for item in result[
            "abnormal_results"
        ]:

            print(
                f"• {item['message']}"
            )

    else:

        print(
            "No abnormal values detected."
        )


    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    print(
        "\nSEVERITY:"
    )

    print(
        result["severity"]
    )


    print(
        "\nCLINICAL FLAG:"
    )

    print(
        result[
            "clinical_flag_message"
        ]
    )


    # --------------------------------------------------------
    # Follow-up
    # --------------------------------------------------------

    print(
        "\nFOLLOW-UP:"
    )

    for item in result[
        "follow_up"
    ]:

        print(
            f"• {item}"
        )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\nPATTERN SUMMARY:"
    )

    print(
        result[
            "pattern_summary"
        ]
    )


    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    output_file = (
        "cbc_interpretation.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2
        )


    print(
        f"\n✅ Saved:"
        f"\n   {output_file}"
    )

    print(
        "\n⚠️ Prototype interpretation only."
        "\nReplace reference ranges and alert rules"
        "\nwith validated clinical/laboratory criteria"
        "\nbefore any real clinical use."
    )