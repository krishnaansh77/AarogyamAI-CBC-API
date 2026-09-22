# ============================================================
# AAROGYAM AI
# CBC MULTI-TASK ENSEMBLE TRAINING
# ============================================================
#
# DATA:
#   cbc_training_data_v2.csv
#
# PRIMARY TARGET:
#   Primary_Label
#
# SECONDARY TARGETS:
#   Dengue_signal
#   Malaria_signal
#   Iron_deficiency_signal
#   Megaloblastic_signal
#   Hemolytic_signal
#
# MODELS:
#   XGBoost
#   LightGBM
#
# ENSEMBLE:
#   Validation-optimized weighted probability averaging
#
# IMPORTANT:
#   - No preprocessing leakage
#   - Test set never used for tuning
#   - No SMOTE on validation/test
#   - Tree models do not require StandardScaler
#   - Synthetic-data prototype only
# ============================================================

import os
import datetime
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# GLOBAL SETTINGS
# ============================================================

SEED = 42

np.random.seed(SEED)

DATA_PATH = "cbc_training_data_v2.csv"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TIMESTAMP = datetime.datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    f"cbc_multitask_ensemble_{TIMESTAMP}.pkl"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    MODEL_DIR,
    f"cbc_feature_importance_{TIMESTAMP}.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Age",
    "Height",
    "Weight",
    "BMI",

    "Hb",
    "RBC",
    "WBC",
    "Platelets",

    "Neutrophils",
    "Lymphocytes",
    "Monocytes",
    "Eosinophils",
    "Basophils",

    "MCV",
    "MCH",
    "MCHC",
    "RDW"
]


# ============================================================
# PRIMARY LABELS
# ============================================================

PRIMARY_LABELS = {
    0: "Normal CBC pattern",
    1: "Anemia-associated pattern",
    2: "Infection-associated pattern",
    3: "Inflammatory pattern",
    4: "Thrombocytopenia pattern",
    5: "Leukopenia pattern",
    6: "Leukocytosis pattern",
    7: "Pancytopenia pattern",
    8: "Erythrocytosis pattern",
    9: "Other abnormal CBC pattern"
}


# ============================================================
# SECONDARY SIGNAL TARGETS
# ============================================================

SIGNAL_TARGETS = [
    "Dengue_signal",
    "Malaria_signal",
    "Iron_deficiency_signal",
    "Megaloblastic_signal",
    "Hemolytic_signal"
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    np.random.seed(seed)


set_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("AAROGYAM AI — CBC MULTI-TASK ENSEMBLE TRAINING")
print("=" * 75)

print("\n📂 Loading dataset...")

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

print(
    f"✅ Dataset loaded: "
    f"{df.shape[0]:,} rows × {df.shape[1]} columns"
)


# ============================================================
# VERIFY DATASET
# ============================================================

required_columns = (
    FEATURES
    + ["Primary_Label", "Primary_Label_Name"]
    + SIGNAL_TARGETS
)

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n🔎 Dataset validation")

print("\nMissing values in features:")

print(
    df[FEATURES]
    .isnull()
    .sum()
)


# ============================================================
# FEATURE MATRIX
# ============================================================

X = df[FEATURES].copy()

y_primary = (
    df["Primary_Label"]
    .astype(int)
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================
#
# 70% training
# 15% validation
# 15% final test
#
# TEST IS NEVER USED FOR MODEL SELECTION.
# ============================================================

print("\n✂️ Splitting dataset...")

X_train_val, X_test, \
y_primary_train_val, y_primary_test = train_test_split(

    X,
    y_primary,

    test_size=0.15,

    random_state=SEED,

    stratify=y_primary
)


X_train, X_val, \
y_primary_train, y_primary_val = train_test_split(

    X_train_val,
    y_primary_train_val,

    test_size=0.1764705882,

    random_state=SEED,

    stratify=y_primary_train_val
)


print(
    f"Training:   {len(X_train):,}"
)

print(
    f"Validation: {len(X_val):,}"
)

print(
    f"Test:       {len(X_test):,}"
)


# ============================================================
# IMPUTATION
# ============================================================
#
# FIT ONLY ON TRAINING DATA.
# ============================================================

print("\n🧹 Fitting median imputer...")

primary_imputer = SimpleImputer(
    strategy="median"
)

X_train_imp = primary_imputer.fit_transform(
    X_train
)

X_val_imp = primary_imputer.transform(
    X_val
)

X_test_imp = primary_imputer.transform(
    X_test
)

print("✅ Imputer fitted on training data only")


# ============================================================
# CLASS WEIGHTS FOR PRIMARY TASK
# ============================================================

def make_multiclass_weights(y):
    """
    Balanced inverse-frequency sample weights.
    """

    values, counts = np.unique(
        y,
        return_counts=True
    )

    total = len(y)
    n_classes = len(values)

    weight_map = {
        int(cls): total / (n_classes * count)
        for cls, count
        in zip(values, counts)
    }

    weights = np.array([
        weight_map[int(v)]
        for v in y
    ])

    return weights


primary_sample_weights = make_multiclass_weights(
    y_primary_train.to_numpy()
)


# ============================================================
# BUILD PRIMARY MODELS
# ============================================================

print("\n" + "=" * 75)
print("TRAINING PRIMARY MODELS")
print("=" * 75)


# ------------------------------------------------------------
# XGBOOST
# ------------------------------------------------------------

print("\n🚀 Training XGBoost...")

xgb_primary = XGBClassifier(
    objective="multi:softprob",

    num_class=len(PRIMARY_LABELS),

    n_estimators=500,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.85,

    colsample_bytree=0.85,

    min_child_weight=2,

    gamma=0.1,

    reg_alpha=0.1,

    reg_lambda=1.5,

    eval_metric="mlogloss",

    tree_method="hist",

    random_state=SEED,

    n_jobs=-1
)


xgb_primary.fit(
    X_train_imp,
    y_primary_train,

    sample_weight=primary_sample_weights,

    eval_set=[
        (X_val_imp, y_primary_val)
    ],

    verbose=False
)

print("✅ XGBoost trained")


# ------------------------------------------------------------
# LIGHTGBM
# ------------------------------------------------------------

print("\n🚀 Training LightGBM...")

lgb_primary = LGBMClassifier(

    objective="multiclass",

    num_class=len(PRIMARY_LABELS),

    n_estimators=500,

    learning_rate=0.05,

    max_depth=7,

    num_leaves=31,

    subsample=0.85,

    colsample_bytree=0.85,

    min_child_samples=25,

    reg_alpha=0.1,

    reg_lambda=1.5,

    random_state=SEED,

    n_jobs=-1,

    verbosity=-1
)


lgb_primary.fit(
    X_train_imp,
    y_primary_train,

    sample_weight=primary_sample_weights,

    eval_set=[
        (X_val_imp, y_primary_val)
    ]
)

print("✅ LightGBM trained")


# ============================================================
# WEIGHT SEARCH
# ============================================================

def optimize_multiclass_weight(
    y_true,
    xgb_prob,
    lgb_prob,
    metric="macro_f1"
):
    """
    Search XGBoost weight from 0.0 to 1.0.

    Final:
        ensemble = w * XGB + (1-w) * LightGBM
    """

    best_weight = 0.5

    best_score = -np.inf

    all_results = []

    for w in np.arange(
        0.0,
        1.01,
        0.05
    ):

        ensemble_prob = (
            w * xgb_prob
            + (1.0 - w) * lgb_prob
        )

        # Normalize to protect against numerical issues
        ensemble_prob = (
            ensemble_prob
            /
            ensemble_prob.sum(
                axis=1,
                keepdims=True
            )
        )

        pred = np.argmax(
            ensemble_prob,
            axis=1
        )

        if metric == "macro_f1":

            score = f1_score(
                y_true,
                pred,
                average="macro",
                zero_division=0
            )

        elif metric == "balanced_accuracy":

            score = balanced_accuracy_score(
                y_true,
                pred
            )

        else:
            raise ValueError(
                "Unknown multiclass metric"
            )

        all_results.append(
            (w, score)
        )

        if score > best_score:

            best_score = score

            best_weight = w

    return (
        best_weight,
        1.0 - best_weight,
        best_score,
        all_results
    )


# ============================================================
# PRIMARY VALIDATION PROBABILITIES
# ============================================================

xgb_primary_val_prob = (
    xgb_primary.predict_proba(
        X_val_imp
    )
)

lgb_primary_val_prob = (
    lgb_primary.predict_proba(
        X_val_imp
    )
)


# ============================================================
# FIND BEST PRIMARY WEIGHT
# ============================================================

print("\n⚖️ Optimizing primary ensemble weights...")

(
    primary_xgb_weight,
    primary_lgb_weight,
    primary_best_score,
    primary_weight_results
) = optimize_multiclass_weight(
    y_primary_val,
    xgb_primary_val_prob,
    lgb_primary_val_prob,
    metric="macro_f1"
)


print(
    f"\n✅ Best primary weights:"
)

print(
    f"   XGBoost : "
    f"{primary_xgb_weight:.2f}"
)

print(
    f"   LightGBM: "
    f"{primary_lgb_weight:.2f}"
)

print(
    f"   Validation Macro F1: "
    f"{primary_best_score:.4f}"
)


# ============================================================
# PRIMARY VALIDATION ENSEMBLE
# ============================================================

primary_val_prob = (
    primary_xgb_weight
    * xgb_primary_val_prob
    +
    primary_lgb_weight
    * lgb_primary_val_prob
)

primary_val_prob = (
    primary_val_prob
    /
    primary_val_prob.sum(
        axis=1,
        keepdims=True
    )
)

primary_val_pred = np.argmax(
    primary_val_prob,
    axis=1
)


print("\nPRIMARY VALIDATION PERFORMANCE")

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_primary_val,
            primary_val_pred
        ),
        4
    )
)

print(
    "Balanced Accuracy:",
    round(
        balanced_accuracy_score(
            y_primary_val,
            primary_val_pred
        ),
        4
    )
)

print(
    "Macro F1:",
    round(
        f1_score(
            y_primary_val,
            primary_val_pred,
            average="macro",
            zero_division=0
        ),
        4
    )
)

print(
    "Log Loss:",
    round(
        log_loss(
            y_primary_val,
            primary_val_prob
        ),
        4
    )
)


# ============================================================
# DISEASE SIGNAL TRAINING
# ============================================================

signal_models = {}

signal_metrics = {}

signal_weights = {}


print("\n" + "=" * 75)
print("TRAINING DISEASE-ASSOCIATED SIGNAL MODELS")
print("=" * 75)


for signal_name in SIGNAL_TARGETS:

    print(
        f"\n🔬 {signal_name}"
    )

    y_signal = (
        df[signal_name]
        .astype(int)
    )


    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    X_sig_train_val, X_sig_test, \
    y_sig_train_val, y_sig_test = train_test_split(

        X,
        y_signal,

        test_size=0.15,

        random_state=SEED,

        stratify=y_signal
    )


    X_sig_train, X_sig_val, \
    y_sig_train, y_sig_val = train_test_split(

        X_sig_train_val,
        y_sig_train_val,

        test_size=0.1764705882,

        random_state=SEED,

        stratify=y_sig_train_val
    )


    # --------------------------------------------------------
    # Imputer
    # --------------------------------------------------------

    signal_imputer = SimpleImputer(
        strategy="median"
    )


    X_sig_train_imp = (
        signal_imputer.fit_transform(
            X_sig_train
        )
    )

    X_sig_val_imp = (
        signal_imputer.transform(
            X_sig_val
        )
    )

    X_sig_test_imp = (
        signal_imputer.transform(
            X_sig_test
        )
    )


    # --------------------------------------------------------
    # Positive class weighting
    # --------------------------------------------------------

    positive = int(
        y_sig_train.sum()
    )

    negative = int(
        len(y_sig_train)
        - positive
    )

    pos_weight = (
        negative / positive
        if positive > 0
        else 1.0
    )


    print(
        f"Positive: {positive:,}"
    )

    print(
        f"Negative: {negative:,}"
    )

    print(
        f"Scale positive weight: "
        f"{pos_weight:.2f}"
    )


    # --------------------------------------------------------
    # XGBOOST SIGNAL MODEL
    # --------------------------------------------------------

    xgb_signal = XGBClassifier(

        objective="binary:logistic",

        n_estimators=400,

        max_depth=5,

        learning_rate=0.05,

        subsample=0.85,

        colsample_bytree=0.85,

        min_child_weight=2,

        gamma=0.1,

        reg_alpha=0.1,

        reg_lambda=1.5,

        eval_metric="logloss",

        tree_method="hist",

        scale_pos_weight=pos_weight,

        random_state=SEED,

        n_jobs=-1
    )


    xgb_signal.fit(
        X_sig_train_imp,
        y_sig_train,

        eval_set=[
            (
                X_sig_val_imp,
                y_sig_val
            )
        ],

        verbose=False
    )


    # --------------------------------------------------------
    # LIGHTGBM SIGNAL MODEL
    # --------------------------------------------------------

    lgb_signal = LGBMClassifier(

        objective="binary",

        n_estimators=400,

        learning_rate=0.05,

        max_depth=6,

        num_leaves=31,

        subsample=0.85,

        colsample_bytree=0.85,

        min_child_samples=25,

        reg_alpha=0.1,

        reg_lambda=1.5,

        scale_pos_weight=pos_weight,

        random_state=SEED,

        n_jobs=-1,

        verbosity=-1
    )


    lgb_signal.fit(
        X_sig_train_imp,
        y_sig_train
    )


    # --------------------------------------------------------
    # VALIDATION PROBABILITIES
    # --------------------------------------------------------

    xgb_prob = (
        xgb_signal
        .predict_proba(
            X_sig_val_imp
        )[:, 1]
    )

    lgb_prob = (
        lgb_signal
        .predict_proba(
            X_sig_val_imp
        )[:, 1]
    )


    # --------------------------------------------------------
    # OPTIMIZE SIGNAL ENSEMBLE WEIGHT
    # --------------------------------------------------------

    best_weight = 0.5

    best_auc = -np.inf

    weight_results = []


    for w in np.arange(
        0.0,
        1.01,
        0.05
    ):

        ensemble_prob = (
            w * xgb_prob
            +
            (1.0 - w)
            * lgb_prob
        )

        try:

            auc = roc_auc_score(
                y_sig_val,
                ensemble_prob
            )

        except ValueError:

            auc = 0.0


        weight_results.append(
            (w, auc)
        )


        if auc > best_auc:

            best_auc = auc
            best_weight = w


    lgb_weight = (
        1.0 - best_weight
    )


    # --------------------------------------------------------
    # FINAL VALIDATION PROBABILITY
    # --------------------------------------------------------

    ensemble_prob = (
        best_weight * xgb_prob
        +
        lgb_weight * lgb_prob
    )


    ensemble_pred = (
        ensemble_prob >= 0.5
    ).astype(int)


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    auc = roc_auc_score(
        y_sig_val,
        ensemble_prob
    )

    pr_auc = average_precision_score(
        y_sig_val,
        ensemble_prob
    )

    f1 = f1_score(
        y_sig_val,
        ensemble_pred,
        zero_division=0
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    signal_models[signal_name] = {

        "imputer": signal_imputer,

        "xgb_model": xgb_signal,

        "lgb_model": lgb_signal,

        "xgb_weight": best_weight,

        "lgb_weight": lgb_weight,

        "positive_class_weight":
            pos_weight
    }


    signal_weights[signal_name] = {

        "xgboost": best_weight,

        "lightgbm": lgb_weight
    }


    signal_metrics[signal_name] = {

        "validation_roc_auc": auc,

        "validation_pr_auc": pr_auc,

        "validation_f1": f1
    }


    print(
        f"XGBoost weight : "
        f"{best_weight:.2f}"
    )

    print(
        f"LightGBM weight: "
        f"{lgb_weight:.2f}"
    )

    print(
        f"ROC-AUC        : "
        f"{auc:.4f}"
    )

    print(
        f"PR-AUC         : "
        f"{pr_auc:.4f}"
    )

    print(
        f"F1             : "
        f"{f1:.4f}"
    )


# ============================================================
# FINAL PRIMARY TEST EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("FINAL PRIMARY TEST EVALUATION")
print("=" * 75)


xgb_test_prob = (
    xgb_primary
    .predict_proba(
        X_test_imp
    )
)

lgb_test_prob = (
    lgb_primary
    .predict_proba(
        X_test_imp
    )
)


primary_test_prob = (
    primary_xgb_weight
    * xgb_test_prob
    +
    primary_lgb_weight
    * lgb_test_prob
)


# Numerical normalization
primary_test_prob = (
    primary_test_prob
    /
    primary_test_prob.sum(
        axis=1,
        keepdims=True
    )
)


primary_test_pred = np.argmax(
    primary_test_prob,
    axis=1
)


print("\nClassification Report:\n")

print(
    classification_report(
        y_primary_test,
        primary_test_pred,

        labels=list(
            PRIMARY_LABELS.keys()
        ),

        target_names=list(
            PRIMARY_LABELS.values()
        ),

        digits=4,

        zero_division=0
    )
)


print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_primary_test,
        primary_test_pred
    )
)


test_accuracy = accuracy_score(
    y_primary_test,
    primary_test_pred
)

test_balanced_accuracy = (
    balanced_accuracy_score(
        y_primary_test,
        primary_test_pred
    )
)

test_macro_f1 = (
    f1_score(
        y_primary_test,
        primary_test_pred,
        average="macro",
        zero_division=0
    )
)

test_log_loss = (
    log_loss(
        y_primary_test,
        primary_test_prob
    )
)


print(
    f"\nAccuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{test_balanced_accuracy:.4f}"
)

print(
    f"Macro F1: "
    f"{test_macro_f1:.4f}"
)

print(
    f"Log Loss: "
    f"{test_log_loss:.4f}"
)


# ============================================================
# DISEASE SIGNAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("DISEASE SIGNAL TEST EVALUATION")
print("=" * 75)


for signal_name in SIGNAL_TARGETS:

    print(
        f"\n🔬 {signal_name}"
    )

    y_signal = (
        df[signal_name]
        .astype(int)
    )


    # Re-create the same deterministic split
    X_sig_train_val, X_sig_test, \
    y_sig_train_val, y_sig_test = train_test_split(

        X,
        y_signal,

        test_size=0.15,

        random_state=SEED,

        stratify=y_signal
    )


    model_info = signal_models[
        signal_name
    ]


    signal_imputer = model_info[
        "imputer"
    ]


    X_sig_test_imp = (
        signal_imputer.transform(
            X_sig_test
        )
    )


    xgb_test_prob = (
        model_info["xgb_model"]
        .predict_proba(
            X_sig_test_imp
        )[:, 1]
    )


    lgb_test_prob = (
        model_info["lgb_model"]
        .predict_proba(
            X_sig_test_imp
        )[:, 1]
    )


    signal_test_prob = (
        model_info["xgb_weight"]
        * xgb_test_prob
        +
        model_info["lgb_weight"]
        * lgb_test_prob
    )


    signal_test_pred = (
        signal_test_prob >= 0.5
    ).astype(int)


    auc = roc_auc_score(
        y_sig_test,
        signal_test_prob
    )


    pr_auc = average_precision_score(
        y_sig_test,
        signal_test_prob
    )


    f1 = f1_score(
        y_sig_test,
        signal_test_pred,
        zero_division=0
    )


    print(
        f"ROC-AUC: {auc:.4f}"
    )

    print(
        f"PR-AUC : {pr_auc:.4f}"
    )

    print(
        f"F1     : {f1:.4f}"
    )


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 75)
print("GLOBAL FEATURE IMPORTANCE")
print("=" * 75)


xgb_imp = (
    xgb_primary
    .feature_importances_
)

lgb_imp = (
    lgb_primary
    .feature_importances_
)


# Normalize both
xgb_imp = (
    xgb_imp /
    np.sum(xgb_imp)
)

lgb_imp = (
    lgb_imp /
    np.sum(lgb_imp)
)


weighted_importance = (
    primary_xgb_weight
    * xgb_imp
    +
    primary_lgb_weight
    * lgb_imp
)


importance_df = pd.DataFrame({

    "Feature": FEATURES,

    "XGBoost_importance":
        xgb_imp,

    "LightGBM_importance":
        lgb_imp,

    "Ensemble_importance":
        weighted_importance
})


importance_df = (
    importance_df
    .sort_values(
        "Ensemble_importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    importance_df.to_string(
        index=False
    )
)


importance_df.to_csv(
    FEATURE_IMPORTANCE_PATH,
    index=False
)


# ============================================================
# SAVE PIPELINE
# ============================================================

pipeline = {

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    "model_name":
        "Aarogyam CBC Multi-Task Ensemble",

    "model_version":
        "3.0",

    "created_at":
        TIMESTAMP,

    "random_seed":
        SEED,

    # --------------------------------------------------------
    # Input features
    # --------------------------------------------------------

    "features":
        FEATURES,

    # --------------------------------------------------------
    # Primary task
    # --------------------------------------------------------

    "primary_labels":
        PRIMARY_LABELS,

    "primary_imputer":
        primary_imputer,

    "primary_xgb_model":
        xgb_primary,

    "primary_lgb_model":
        lgb_primary,

    "primary_ensemble_weights": {

        "xgboost":
            primary_xgb_weight,

        "lightgbm":
            primary_lgb_weight
    },

    # --------------------------------------------------------
    # Disease signals
    # --------------------------------------------------------

    "signal_targets":
        SIGNAL_TARGETS,

    "signal_models":
        signal_models,

    "signal_weights":
        signal_weights,

    "signal_validation_metrics":
        signal_metrics,

    # --------------------------------------------------------
    # Global importance
    # --------------------------------------------------------

    "primary_feature_importance":
        importance_df,

    # --------------------------------------------------------
    # Dataset metadata
    # --------------------------------------------------------

    "training_info": {

        "dataset":
            DATA_PATH,

        "total_samples":
            int(len(df)),

        "feature_count":
            len(FEATURES),

        "primary_class_count":
            len(PRIMARY_LABELS),

        "signal_count":
            len(SIGNAL_TARGETS)
    },

    # --------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------

    "primary_test_metrics": {

        "accuracy":
            float(test_accuracy),

        "balanced_accuracy":
            float(test_balanced_accuracy),

        "macro_f1":
            float(test_macro_f1),

        "log_loss":
            float(test_log_loss)
    }
}


joblib.dump(
    pipeline,
    MODEL_PATH
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("✅ TRAINING COMPLETE")
print("=" * 75)

print(
    f"\nModel saved:"
    f"\n  {MODEL_PATH}"
)

print(
    f"\nFeature importance saved:"
    f"\n  {FEATURE_IMPORTANCE_PATH}"
)

print("\nPRIMARY ENSEMBLE WEIGHTS")

print(
    f"XGBoost : "
    f"{primary_xgb_weight:.2f}"
)

print(
    f"LightGBM: "
    f"{primary_lgb_weight:.2f}"
)

print("\nDISEASE SIGNAL WEIGHTS")

for signal_name, weights in signal_weights.items():

    print(
        f"\n{signal_name}"
    )

    print(
        f"  XGBoost : "
        f"{weights['xgboost']:.2f}"
    )

    print(
        f"  LightGBM: "
        f"{weights['lightgbm']:.2f}"
    )


print(
    "\n⚠️ IMPORTANT:"
    "\nThis model was trained on synthetic data."
    "\nIt is a prototype and must not be treated as"
    "\na clinically validated diagnostic model."
)