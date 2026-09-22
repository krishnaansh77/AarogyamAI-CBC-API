# ============================================================
# AAROGYAM AI
# CBC PATIENT-SPECIFIC EXPLANATION
# ============================================================
#
# Loads the trained multi-task ensemble and explains one
# patient's prediction using SHAP.
#
# OUTPUT:
#   - Primary CBC prediction
#   - Primary probabilities
#   - Disease-associated signal probabilities
#   - Patient-specific feature contributions
#   - Direction of contribution
#
# IMPORTANT:
#   Current model is trained on synthetic data.
#   This is a prototype explanation system.
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = (
    "./models/"
    "cbc_multitask_ensemble_20260826_010538.pkl"
)

TOP_FEATURES = 8
TOP_SIGNALS = 5


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("AAROGYAM AI — CBC PATIENT EXPLANATION")
print("=" * 70)

print("\n📦 Loading trained model...")

pipeline = joblib.load(MODEL_PATH)

print("✅ Model loaded")


# ============================================================
# LOAD PIPELINE COMPONENTS
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

SIGNAL_MODELS = pipeline["signal_models"]


# ============================================================
# PATIENT INPUT
# ============================================================
#
# Replace these values when testing another patient.
# ============================================================

patient = {
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


# ============================================================
# CREATE DATAFRAME
# ============================================================

patient_df = pd.DataFrame(
    [patient],
    columns=FEATURES
)


# ============================================================
# IMPUTE
# ============================================================

patient_imputed = PRIMARY_IMPUTER.transform(
    patient_df
)


# ============================================================
# SHAP HELPER
# ============================================================

def get_single_class_shap(
    model,
    sample,
    class_index=None
):
    """
    Return SHAP values for a single observation.

    Works with tree-based XGBoost/LightGBM models.
    """

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        sample
    )

    # ------------------------------
    # Binary model
    # ------------------------------

    if class_index is None:

        if isinstance(shap_values, list):

            values = np.asarray(
                shap_values[0]
            )

        else:

            values = np.asarray(
                shap_values
            )

        values = values.reshape(-1)

        return values


    # ------------------------------
    # Multiclass model
    # ------------------------------

    if isinstance(shap_values, list):

        values = np.asarray(
            shap_values[class_index]
        )

        return values.reshape(-1)


    shap_values = np.asarray(
        shap_values
    )

    # Common format:
    # (samples, features, classes)

    if shap_values.ndim == 3:

        return shap_values[
            0,
            :,
            class_index
        ]


    # Fallback
    if shap_values.ndim == 2:

        return shap_values[
            0
        ]

    raise ValueError(
        "Unexpected SHAP output shape: "
        f"{shap_values.shape}"
    )


# ============================================================
# PRIMARY PREDICTION
# ============================================================

xgb_primary_prob = (
    PRIMARY_XGB
    .predict_proba(
        patient_imputed
    )[0]
)

lgb_primary_prob = (
    PRIMARY_LGB
    .predict_proba(
        patient_imputed
    )[0]
)


primary_xgb_weight = PRIMARY_WEIGHTS[
    "xgboost"
]

primary_lgb_weight = PRIMARY_WEIGHTS[
    "lightgbm"
]


primary_probability = (
    primary_xgb_weight
    * xgb_primary_prob
    +
    primary_lgb_weight
    * lgb_primary_prob
)


# Numerical normalization
primary_probability = (
    primary_probability
    /
    primary_probability.sum()
)


primary_class_index = int(
    np.argmax(
        primary_probability
    )
)

primary_class_name = PRIMARY_LABELS[
    primary_class_index
]

primary_confidence = float(
    primary_probability[
        primary_class_index
    ]
)


# ============================================================
# PRIMARY SHAP
# ============================================================

xgb_primary_shap = get_single_class_shap(
    PRIMARY_XGB,
    patient_imputed,
    primary_class_index
)

lgb_primary_shap = get_single_class_shap(
    PRIMARY_LGB,
    patient_imputed,
    primary_class_index
)


# Ensemble SHAP contribution

primary_shap = (
    primary_xgb_weight
    * xgb_primary_shap
    +
    primary_lgb_weight
    * lgb_primary_shap
)


# ============================================================
# CREATE PRIMARY CONTRIBUTIONS
# ============================================================

primary_contributions = []

for i, feature in enumerate(FEATURES):

    shap_value = float(
        primary_shap[i]
    )

    patient_value = patient[
        feature
    ]

    # Positive = pushing toward prediction
    # Negative = pushing away from prediction

    if shap_value > 0:
        direction = "toward_prediction"
    elif shap_value < 0:
        direction = "away_from_prediction"
    else:
        direction = "neutral"

    primary_contributions.append({

        "parameter": feature,

        "value": patient_value,

        "shap_value": shap_value,

        "absolute_importance": abs(
            shap_value
        ),

        "direction": direction
    })


primary_contributions.sort(
    key=lambda x: x[
        "absolute_importance"
    ],
    reverse=True
)


top_primary_contributions = (
    primary_contributions[
        :TOP_FEATURES
    ]
)


# ============================================================
# SIGNAL MODELS
# ============================================================

signal_results = []


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


    # --------------------------------------------------------
    # Transform patient with signal-specific imputer
    # --------------------------------------------------------

    signal_patient = (
        signal_imputer.transform(
            patient_df
        )
    )


    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    xgb_prob = (
        xgb_model
        .predict_proba(
            signal_patient
        )[0, 1]
    )

    lgb_prob = (
        lgb_model
        .predict_proba(
            signal_patient
        )[0, 1]
    )

    ensemble_prob = (
        xgb_weight
        * xgb_prob
        +
        lgb_weight
        * lgb_prob
    )


    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    xgb_shap = get_single_class_shap(
        xgb_model,
        signal_patient
    )

    lgb_shap = get_single_class_shap(
        lgb_model,
        signal_patient
    )


    ensemble_shap = (
        xgb_weight * xgb_shap
        +
        lgb_weight * lgb_shap
    )


    # --------------------------------------------------------
    # Factor list
    # --------------------------------------------------------

    factors = []

    for i, feature in enumerate(FEATURES):

        shap_value = float(
            ensemble_shap[i]
        )

        if shap_value > 0:

            direction = (
                "toward_signal"
            )

        elif shap_value < 0:

            direction = (
                "away_from_signal"
            )

        else:

            direction = "neutral"


        factors.append({

            "parameter": feature,

            "value": patient[
                feature
            ],

            "shap_value": shap_value,

            "absolute_importance": abs(
                shap_value
            ),

            "direction": direction
        })


    factors.sort(
        key=lambda x: x[
            "absolute_importance"
        ],
        reverse=True
    )


    signal_results.append({

        "name": signal_name,

        "probability": float(
            ensemble_prob
        ),

        "xgboost_weight": float(
            xgb_weight
        ),

        "lightgbm_weight": float(
            lgb_weight
        ),

        "contributing_factors":
            factors[:TOP_FEATURES]
    })


# ============================================================
# SORT SIGNALS
# ============================================================

signal_results.sort(
    key=lambda x: x[
        "probability"
    ],
    reverse=True
)


# ============================================================
# TOP CBC PROBABILITIES
# ============================================================

primary_predictions = []

for index, probability in enumerate(
    primary_probability
):

    primary_predictions.append({

        "name":
            PRIMARY_LABELS[index],

        "probability":
            float(probability)
    })


primary_predictions.sort(
    key=lambda x: x[
        "probability"
    ],
    reverse=True
)


# ============================================================
# RESULT OBJECT
# ============================================================

result = {

    "primary_prediction": {

        "name":
            primary_class_name,

        "probability":
            primary_confidence,

        "confidence_level":
            (
                "high"
                if primary_confidence >= 0.70
                else
                "moderate"
                if primary_confidence >= 0.40
                else
                "low"
            )
    },


    "primary_predictions":
        primary_predictions,


    "disease_associated_signals":
        signal_results,


    "contributing_factors":
        top_primary_contributions,


    "model_info": {

        "name":
            pipeline["model_name"],

        "version":
            pipeline["model_version"]
    }
}


# ============================================================
# PRINT RESULT
# ============================================================

print("\n" + "=" * 70)
print("PRIMARY CBC PREDICTION")
print("=" * 70)

print(
    f"\n{primary_class_name}"
)

print(
    f"Probability: "
    f"{primary_confidence:.2%}"
)


# ------------------------------------------------------------
# Primary probabilities
# ------------------------------------------------------------

print("\nPRIMARY CBC PATTERNS")

for item in primary_predictions:

    print(
        f"{item['name']:<40}"
        f"{item['probability']:.2%}"
    )


# ------------------------------------------------------------
# Primary contributing factors
# ------------------------------------------------------------

print(
    "\nWHY THIS PRIMARY PREDICTION?"
)

for factor in top_primary_contributions:

    print(
        f"{factor['parameter']:<18}"
        f"value={str(factor['value']):<10}"
        f"SHAP={factor['shap_value']:+.4f}"
        f"  {factor['direction']}"
    )


# ------------------------------------------------------------
# Disease signals
# ------------------------------------------------------------

print(
    "\nDISEASE-ASSOCIATED SIGNALS"
)

for signal in signal_results:

    print(
        f"\n{signal['name']}"
        f" → {signal['probability']:.2%}"
    )

    print(
        f"   XGB weight: "
        f"{signal['xgboost_weight']:.2f}"
    )

    print(
        f"   LGB weight: "
        f"{signal['lightgbm_weight']:.2f}"
    )

    print(
        "   Top contributing factors:"
    )

    for factor in signal[
        "contributing_factors"
    ][:5]:

        print(
            f"      "
            f"{factor['parameter']:<18}"
            f"{factor['shap_value']:+.4f}"
            f"  {factor['direction']}"
        )


# ============================================================
# SAVE JSON RESULT
# ============================================================

OUTPUT_FILE = (
    "cbc_patient_explanation.json"
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        indent=2
    )


print(
    f"\n✅ Explanation JSON saved:"
    f"\n   {OUTPUT_FILE}"
)

print(
    "\n⚠️ This explanation is generated from a"
    "\nsynthetic-data prototype and is not a"
    "\nclinical diagnosis."
)