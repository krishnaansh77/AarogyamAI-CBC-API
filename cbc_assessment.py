# ============================================================
# AAROGYAM AI
# CBC COMPLETE ASSESSMENT ENGINE
# ============================================================
#
# Combines:
#   1. Trained CBC ML ensemble
#   2. Patient-specific SHAP explanations
#   3. CBC reference-range interpretation
#   4. Severity / follow-up layer
#
# INPUT:
#   One patient's CBC dictionary
#
# OUTPUT:
#   One frontend-ready JSON-compatible dictionary
#
# IMPORTANT:
#   Current model + interpretation use synthetic/prototype data.
#   This is NOT a clinically validated diagnostic system.
# ============================================================

import json
import os

import joblib
import numpy as np
import pandas as pd
import shap

from cbc_interpreter import build_interpretation


# ============================================================
# MODEL PATH
# ============================================================
#
# Update this automatically when you create a newer model.
# ============================================================

MODEL_PATH = (
    "./models/"
    "cbc_multitask_ensemble_20260826_010538.pkl"
)


# ============================================================
# DISPLAY SETTINGS
# ============================================================

TOP_PRIMARY_PREDICTIONS = 5

TOP_DISEASE_SIGNALS = 5

DISEASE_DISPLAY_THRESHOLD = 0.05

TOP_CONTRIBUTING_FACTORS = 8


# ============================================================
# LOAD MODEL
# ============================================================

print("\n📦 Loading CBC assessment model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}"
    )

pipeline = joblib.load(MODEL_PATH)

print("✅ Model loaded")


# ============================================================
# LOAD MODEL COMPONENTS
# ============================================================

FEATURES = pipeline["features"]

PRIMARY_LABELS = pipeline["primary_labels"]

SIGNAL_TARGETS = pipeline["signal_targets"]

PRIMARY_IMPUTER = pipeline["primary_imputer"]

PRIMARY_XGB = pipeline["primary_xgb_model"]

PRIMARY_LGB = pipeline["primary_lgb_model"]

PRIMARY_WEIGHTS = pipeline[
    "primary_ensemble_weights"
]

SIGNAL_MODELS = pipeline[
    "signal_models"
]

MODEL_NAME = pipeline[
    "model_name"
]

MODEL_VERSION = pipeline[
    "model_version"
]


# ============================================================
# SHAP HELPER
# ============================================================

def get_shap_values(
    model,
    sample,
    class_index=None
):
    """
    Handles common SHAP TreeExplainer outputs for
    XGBoost and LightGBM.

    Returns:
        1D numpy array containing one value per feature.
    """

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        sample
    )

    # --------------------------------------------------------
    # Convert lists to numpy arrays
    # --------------------------------------------------------

    if isinstance(shap_values, list):

        arrays = [
            np.asarray(value)
            for value in shap_values
        ]

        # Binary model:
        # LightGBM may return a one-element list
        # containing the positive-class SHAP values.
        if class_index is None:

            if len(arrays) == 1:
                return arrays[0].reshape(-1)

            # If two classes are returned, class 1 is
            # normally the positive class.
            return arrays[-1].reshape(-1)

        # Multiclass model
        return arrays[class_index].reshape(-1)

    # --------------------------------------------------------
    # Array output
    # --------------------------------------------------------

    shap_array = np.asarray(
        shap_values
    )

    # Binary/single-output:
    #
    # (samples, features)
    if shap_array.ndim == 2:

        return shap_array[0].reshape(-1)

    # Multiclass:
    #
    # Common format:
    # (samples, features, classes)
    if shap_array.ndim == 3:

        if class_index is None:
            raise ValueError(
                "class_index is required for "
                "multiclass SHAP output."
            )

        return shap_array[
            0,
            :,
            class_index
        ].reshape(-1)

    raise ValueError(
        "Unexpected SHAP shape: "
        f"{shap_array.shape}"
    )


# ============================================================
# NORMALIZE CONTRIBUTIONS FOR UI
# ============================================================

def normalize_contributions(
    shap_values
):
    """
    Convert absolute SHAP values into relative
    importance percentages for UI display.

    IMPORTANT:
        This is only a relative visualization score.
        It is NOT a probability and NOT a clinical
        contribution percentage.
    """

    absolute_values = np.abs(
        np.asarray(shap_values)
    )

    total = absolute_values.sum()

    if total == 0:

        return np.zeros_like(
            absolute_values,
            dtype=float
        )

    return (
        absolute_values / total
    )


# ============================================================
# PRIMARY MODEL PREDICTION
# ============================================================

def get_primary_prediction(
    patient_df
):
    """
    Returns:
        Primary probabilities
        Top prediction
        SHAP-based contributing factors
    """

    transformed = PRIMARY_IMPUTER.transform(
        patient_df
    )

    # --------------------------------------------------------
    # Base probabilities
    # --------------------------------------------------------

    xgb_prob = PRIMARY_XGB.predict_proba(
        transformed
    )[0]

    lgb_prob = PRIMARY_LGB.predict_proba(
        transformed
    )[0]

    # --------------------------------------------------------
    # Learned ensemble weights
    # --------------------------------------------------------

    xgb_weight = PRIMARY_WEIGHTS[
        "xgboost"
    ]

    lgb_weight = PRIMARY_WEIGHTS[
        "lightgbm"
    ]

    probabilities = (
        xgb_weight * xgb_prob
        +
        lgb_weight * lgb_prob
    )

    # Numerical normalization
    probabilities = (
        probabilities
        /
        probabilities.sum()
    )

    # --------------------------------------------------------
    # Primary class
    # --------------------------------------------------------

    best_index = int(
        np.argmax(probabilities)
    )

    best_name = PRIMARY_LABELS[
        best_index
    ]

    best_probability = float(
        probabilities[best_index]
    )

    # --------------------------------------------------------
    # Confidence label
    # --------------------------------------------------------

    if best_probability >= 0.70:
        confidence_level = "high"

    elif best_probability >= 0.40:
        confidence_level = "moderate"

    else:
        confidence_level = "low"

    # --------------------------------------------------------
    # All primary predictions
    # --------------------------------------------------------

    all_predictions = []

    for index, probability in enumerate(
        probabilities
    ):

        all_predictions.append({

            "name":
                PRIMARY_LABELS[index],

            "probability":
                float(probability)
        })

    all_predictions.sort(
        key=lambda x: x["probability"],
        reverse=True
    )

    # --------------------------------------------------------
    # SHAP explanation for winning class
    # --------------------------------------------------------

    xgb_shap = get_shap_values(
        PRIMARY_XGB,
        transformed,
        class_index=best_index
    )

    lgb_shap = get_shap_values(
        PRIMARY_LGB,
        transformed,
        class_index=best_index
    )

    ensemble_shap = (
        xgb_weight * xgb_shap
        +
        lgb_weight * lgb_shap
    )

    relative_importance = (
        normalize_contributions(
            ensemble_shap
        )
    )

    factors = []

    for i, feature in enumerate(
        FEATURES
    ):

        shap_value = float(
            ensemble_shap[i]
        )

        factors.append({

            "parameter":
                feature,

            "value":
                patient_df.iloc[0][feature],

            "shap_value":
                shap_value,

            "importance":
                float(
                    relative_importance[i]
                ),

            "direction":
                (
                    "toward_prediction"
                    if shap_value > 0
                    else
                    "away_from_prediction"
                    if shap_value < 0
                    else
                    "neutral"
                )
        })

    factors.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    return {

        "primary_prediction": {

            "name":
                best_name,

            "probability":
                best_probability,

            "confidence_level":
                confidence_level
        },

        "alternative_predictions": (
            all_predictions[
                1:TOP_PRIMARY_PREDICTIONS
            ]
        ),

        "all_primary_predictions":
            all_predictions,

        "contributing_factors":
            factors[
                :TOP_CONTRIBUTING_FACTORS
            ]
    }


# ============================================================
# DISEASE-ASSOCIATED SIGNALS
# ============================================================

def get_disease_signals(
    patient_df
):
    """
    Run the five independent disease-associated
    signal models.

    These are signals/patterns, NOT diagnoses.
    """

    signals = []

    for signal_name in SIGNAL_TARGETS:

        model_info = SIGNAL_MODELS[
            signal_name
        ]

        signal_imputer = model_info[
            "imputer"
        ]

        xgb_model = model_info[
            "xgb_model"
        ]

        lgb_model = model_info[
            "lgb_model"
        ]

        xgb_weight = model_info[
            "xgb_weight"
        ]

        lgb_weight = model_info[
            "lgb_weight"
        ]

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        transformed = (
            signal_imputer.transform(
                patient_df
            )
        )

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        xgb_probability = (
            xgb_model.predict_proba(
                transformed
            )[0, 1]
        )

        lgb_probability = (
            lgb_model.predict_proba(
                transformed
            )[0, 1]
        )

        probability = (
            xgb_weight
            * xgb_probability
            +
            lgb_weight
            * lgb_probability
        )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        xgb_shap = get_shap_values(
            xgb_model,
            transformed
        )

        lgb_shap = get_shap_values(
            lgb_model,
            transformed
        )

        ensemble_shap = (
            xgb_weight * xgb_shap
            +
            lgb_weight * lgb_shap
        )

        relative_importance = (
            normalize_contributions(
                ensemble_shap
            )
        )

        factors = []

        for i, feature in enumerate(
            FEATURES
        ):

            shap_value = float(
                ensemble_shap[i]
            )

            factors.append({

                "parameter":
                    feature,

                "value":
                    patient_df.iloc[0][feature],

                "shap_value":
                    shap_value,

                "importance":
                    float(
                        relative_importance[i]
                    ),

                "direction":
                    (
                        "toward_signal"
                        if shap_value > 0
                        else
                        "away_from_signal"
                        if shap_value < 0
                        else
                        "neutral"
                    )
            })

        factors.sort(
            key=lambda x: x["importance"],
            reverse=True
        )

        signals.append({

            "name":
                signal_name,

            "probability":
                float(probability),

            "ensemble_weights": {

                "xgboost":
                    float(xgb_weight),

                "lightgbm":
                    float(lgb_weight)
            },

            "contributing_factors":
                factors[
                    :TOP_CONTRIBUTING_FACTORS
                ]
        })

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    signals.sort(
        key=lambda x: x["probability"],
        reverse=True
    )

    # --------------------------------------------------------
    # Only meaningful signals for main UI.
    #
    # Keep all signals internally, but UI can use the
    # filtered list.
    # --------------------------------------------------------

    display_signals = [
        signal
        for signal in signals
        if signal["probability"]
        >= DISEASE_DISPLAY_THRESHOLD
    ]

    # Ensure at least top one remains visible.
    if not display_signals and signals:

        display_signals = signals[:1]

    return (
        display_signals,
        signals
    )


# ============================================================
# BUILD COMPLETE ASSESSMENT
# ============================================================

def assess_cbc(
    patient: dict
):
    """
    Main function.

    This is the function that FastAPI will eventually call.
    """

    # --------------------------------------------------------
    # Validate required fields
    # --------------------------------------------------------

    missing = [
        feature
        for feature in FEATURES
        if feature not in patient
    ]

    if missing:

        raise ValueError(
            "Missing CBC input fields: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    patient_df = pd.DataFrame(
        [patient],
        columns=FEATURES
    )

    # --------------------------------------------------------
    # ML prediction + SHAP
    # --------------------------------------------------------

    prediction = get_primary_prediction(
        patient_df
    )

    # --------------------------------------------------------
    # Disease signals
    # --------------------------------------------------------

    display_signals, all_signals = (
        get_disease_signals(
            patient_df
        )
    )

    # --------------------------------------------------------
    # CBC interpretation
    # --------------------------------------------------------

    interpretation = build_interpretation(
        patient
    )

    # --------------------------------------------------------
    # Build frontend-ready structure
    # --------------------------------------------------------

    result = {

        "primary_prediction":
            prediction[
                "primary_prediction"
            ],

        "alternative_predictions":
            prediction[
                "alternative_predictions"
            ],

        "all_primary_predictions":
            prediction[
                "all_primary_predictions"
            ],

        "disease_associated_signals":
            display_signals,

        "all_disease_associated_signals":
            all_signals,

        "contributing_factors":
            prediction[
                "contributing_factors"
            ],

        "pattern_summary":
            interpretation[
                "pattern_summary"
            ],

        "key_pattern": [],

        "pattern_caption":
            "Observed CBC findings contributing "
            "to the assessment",

        "cbc_results":
            interpretation[
                "cbc_results"
            ],

        "grouped_results":
            interpretation[
                "grouped_results"
            ],

        "lineage_overview":
            interpretation[
                "lineage_overview"
            ],

        "supporting_evidence":
            interpretation[
                "supporting_evidence"
            ],

        "abnormal_results":
            interpretation[
                "abnormal_results"
            ],

        "severity":
            interpretation[
                "severity"
            ],

        "clinical_flag_message":
            interpretation[
                "clinical_flag_message"
            ],

        "follow_up":
            interpretation[
                "follow_up"
            ],

        "model_info": {

            "name":
                MODEL_NAME,

            "version":
                MODEL_VERSION
        },

        "disclaimer":
            (
                "AI-generated CBC-based assessment. "
                "This prototype is intended to support "
                "review and is not a definitive diagnosis."
            )
    }

    # --------------------------------------------------------
    # Create key pattern for frontend
    #
    # We use laboratory direction here, NOT SHAP direction.
    # --------------------------------------------------------

    key_pattern = []

    for factor in prediction[
        "contributing_factors"
    ][:5]:

        parameter_key = factor[
            "parameter"
        ]

        cbc_item = next(
            (
                item
                for item
                in interpretation["cbc_results"]
                if item["key"]
                == parameter_key
            ),
            None
        )

        if cbc_item is None:
            continue

        status = cbc_item[
            "status"
        ]

        if status in [
            "low",
            "high",
            "critical"
        ]:

            key_pattern.append({

                "label":
                    cbc_item[
                        "parameter"
                    ],

                "direction":
                    status,

                "value":
                    cbc_item[
                        "value"
                    ],

                "unit":
                    cbc_item[
                        "unit"
                    ]
            })

    result[
        "key_pattern"
    ] = key_pattern

    return result


# ============================================================
# TEST PATIENT
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 75)
    print("AAROGYAM AI — COMPLETE CBC ASSESSMENT TEST")
    print("=" * 75)

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

    # --------------------------------------------------------
    # Run assessment
    # --------------------------------------------------------

    result = assess_cbc(
        test_patient
    )

    # --------------------------------------------------------
    # Print primary prediction
    # --------------------------------------------------------

    primary = result[
        "primary_prediction"
    ]

    print("\nPRIMARY CBC ASSESSMENT")

    print(
        f"\nPrediction: "
        f"{primary['name']}"
    )

    print(
        f"Probability: "
        f"{primary['probability']:.2%}"
    )

    print(
        f"Confidence level: "
        f"{primary['confidence_level']}"
    )

    # --------------------------------------------------------
    # Alternative predictions
    # --------------------------------------------------------

    print(
        "\nALTERNATIVE PREDICTIONS"
    )

    for prediction in result[
        "alternative_predictions"
    ]:

        print(
            f"  {prediction['name']:<40}"
            f"{prediction['probability']:.2%}"
        )

    # --------------------------------------------------------
    # Disease signals
    # --------------------------------------------------------

    print(
        "\nDISEASE-ASSOCIATED SIGNALS"
    )

    for signal in result[
        "disease_associated_signals"
    ]:

        print(
            f"  {signal['name']:<35}"
            f"{signal['probability']:.2%}"
        )

    # --------------------------------------------------------
    # Contributing factors
    # --------------------------------------------------------

    print(
        "\nTOP CONTRIBUTING FACTORS"
    )

    for factor in result[
        "contributing_factors"
    ][:8]:

        print(
            f"  {factor['parameter']:<18}"
            f"value={str(factor['value']):<10}"
            f"importance="
            f"{factor['importance']:.2%}"
            f"  {factor['direction']}"
        )

    # --------------------------------------------------------
    # Key pattern
    # --------------------------------------------------------

    print(
        "\nKEY CBC PATTERN"
    )

    for item in result[
        "key_pattern"
    ]:

        print(
            f"  {item['label']}: "
            f"{item['value']} "
            f"{item['unit']} "
            f"({item['direction']})"
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

    # --------------------------------------------------------
    # Clinical flag
    # --------------------------------------------------------

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
            f"  • {item}"
        )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    output_file = (
        "cbc_complete_assessment.json"
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
        f"\n✅ Complete assessment saved:"
        f"\n   {output_file}"
    )

    print(
        "\n⚠️ Prototype only — "
        "not a clinically validated diagnostic system."
    )