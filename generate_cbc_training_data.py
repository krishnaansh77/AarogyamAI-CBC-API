# ============================================================
# AAROGYAM AI — CBC SYNTHETIC TRAINING DATA GENERATOR
# ============================================================
#
# PURPOSE:
#   Generate synthetic CBC data for a prototype ML pipeline.
#
# OUTPUTS:
#   1. Primary CBC pattern label:
#        0 Normal CBC pattern
#        1 Anemia-associated pattern
#        2 Infection-associated pattern
#        3 Inflammatory pattern
#        4 Thrombocytopenia pattern
#        5 Leukopenia pattern
#        6 Leukocytosis pattern
#        7 Pancytopenia pattern
#        8 Erythrocytosis pattern
#        9 Other abnormal CBC pattern
#
#   2. Disease-associated binary signals:
#        Dengue_signal
#        Malaria_signal
#        Iron_deficiency_signal
#        Megaloblastic_signal
#        Hemolytic_signal
#
# IMPORTANT:
#   These are synthetic labels created from simulated patterns.
#   They must NOT be treated as clinical ground truth.
# ============================================================

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# GLOBAL SETTINGS
# ------------------------------------------------------------

SEED = 42
np.random.seed(SEED)

TOTAL_SAMPLES = 50000
OUTPUT_FILE = "cbc_training_data_v2.csv"

# ------------------------------------------------------------
# PRIMARY CBC PATTERN LABELS
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------

def clipped_normal(mean, sd, low, high):
    value = np.random.normal(mean, sd)
    return np.clip(value, low, high)


def bounded_uniform(low, high):
    return np.random.uniform(low, high)


def add_noise(value, relative_sd=0.03):
    """
    Add measurement/biological variation.
    """
    noise = np.random.normal(1.0, relative_sd)
    return value * noise


def clamp(value, low, high):
    return float(np.clip(value, low, high))


# ------------------------------------------------------------
# DIFFERENTIAL COUNT NORMALIZATION
# ------------------------------------------------------------

def normalize_differential(d):
    """
    Keep differential percentages approximately summing to 100.
    """

    keys = [
        "Neutrophils",
        "Lymphocytes",
        "Monocytes",
        "Eosinophils",
        "Basophils"
    ]

    values = np.array([max(d[k], 0.1) for k in keys])

    total = values.sum()

    if total <= 0:
        values = np.array([60, 30, 6, 3, 1], dtype=float)
        total = values.sum()

    values = values / total * 100

    for key, value in zip(keys, values):
        d[key] = value

    return d


# ============================================================
# BASE HUMAN GENERATION
# ============================================================

def generate_base_person():

    # Include children + adults
    age = np.random.randint(5, 86)

    # Synthetic sex feature is NOT included in the model input.
    # It can still influence the generated distributions.
    sex = np.random.choice(["Male", "Female"])

    # --------------------------------------------------------
    # Height / Weight / BMI
    # --------------------------------------------------------

    if age < 13:
        height = clipped_normal(140, 18, 100, 175)
        weight = clipped_normal(35, 12, 15, 80)

    elif age < 18:
        height = clipped_normal(162, 10, 135, 195)
        weight = clipped_normal(55, 14, 30, 100)

    elif sex == "Male":
        height = clipped_normal(172, 8, 145, 200)
        weight = clipped_normal(73, 14, 40, 140)

    else:
        height = clipped_normal(160, 7, 140, 195)
        weight = clipped_normal(62, 13, 35, 130)

    bmi = weight / ((height / 100) ** 2)

    bmi = clamp(bmi, 12, 45)

    # --------------------------------------------------------
    # Baseline CBC
    # --------------------------------------------------------

    if sex == "Male":
        hb_mean = 14.8
        rbc_mean = 5.1
    else:
        hb_mean = 13.4
        rbc_mean = 4.6

    # Slight age influence
    if age < 13:
        hb_mean -= 1.0
        rbc_mean -= 0.3

    elif age > 65:
        hb_mean -= 0.2

    hb = clipped_normal(hb_mean, 1.1, 7, 19)

    rbc = clipped_normal(rbc_mean, 0.45, 2.5, 7)

    # Hematocrit is generated internally so RBC indices remain
    # reasonably coherent.
    hematocrit = clamp(
        hb * np.random.uniform(2.85, 3.15),
        20,
        60
    )

    mcv = clipped_normal(90, 5, 60, 110)

    mch = (hb * 10) / rbc
    mch = clamp(mch + np.random.normal(0, 1), 15, 38)

    mchc = (hb / hematocrit) * 100
    mchc = clamp(mchc + np.random.normal(0, 0.7), 25, 38)

    rdw = clipped_normal(13.2, 1.2, 10, 25)

    wbc = clipped_normal(7200, 1400, 2500, 20000)

    platelets = clipped_normal(
        250000,
        45000,
        50000,
        600000
    )

    differential = {
        "Neutrophils": clipped_normal(59, 6, 25, 85),
        "Lymphocytes": clipped_normal(31, 6, 10, 70),
        "Monocytes": clipped_normal(6, 1.5, 2, 15),
        "Eosinophils": clipped_normal(3, 1.2, 0, 10),
        "Basophils": clipped_normal(1, 0.4, 0, 3)
    }

    d = {
        "Age": age,
        "Height": height,
        "Weight": weight,
        "BMI": bmi,

        "Hb": hb,
        "RBC": rbc,
        "WBC": wbc,
        "Platelets": platelets,

        "Neutrophils": differential["Neutrophils"],
        "Lymphocytes": differential["Lymphocytes"],
        "Monocytes": differential["Monocytes"],
        "Eosinophils": differential["Eosinophils"],
        "Basophils": differential["Basophils"],

        "MCV": mcv,
        "MCH": mch,
        "MCHC": mchc,
        "RDW": rdw,

        # Internal fields.
        "_Sex": sex,
        "_Hematocrit": hematocrit
    }

    return normalize_differential(d)


# ============================================================
# PRIMARY CBC PATTERN MODIFIERS
# ============================================================

def apply_normal(d):
    return d


def apply_anemia(d):

    anemia_type = np.random.choice(
        ["microcytic", "normocytic", "macrocytic"],
        p=[0.45, 0.35, 0.20]
    )

    d["Hb"] -= bounded_uniform(2.0, 5.0)
    d["RBC"] -= bounded_uniform(0.2, 1.0)
    d["RDW"] += bounded_uniform(1.0, 6.0)

    if anemia_type == "microcytic":
        d["MCV"] -= bounded_uniform(8, 22)
        d["MCH"] -= bounded_uniform(3, 7)
        d["MCHC"] -= bounded_uniform(0.5, 2.5)

    elif anemia_type == "macrocytic":
        d["MCV"] += bounded_uniform(7, 18)
        d["MCH"] += bounded_uniform(1, 4)

    else:
        d["MCV"] += np.random.normal(0, 3)

    return d


def apply_infection(d):

    infection_type = np.random.choice(
        ["bacterial_like", "viral_like", "mixed"],
        p=[0.50, 0.30, 0.20]
    )

    if infection_type == "bacterial_like":

        d["WBC"] += bounded_uniform(2500, 10000)
        d["Neutrophils"] += bounded_uniform(8, 25)
        d["Lymphocytes"] -= bounded_uniform(3, 12)

    elif infection_type == "viral_like":

        d["WBC"] -= bounded_uniform(1000, 3500)
        d["Lymphocytes"] += bounded_uniform(8, 20)
        d["Neutrophils"] -= bounded_uniform(5, 15)

    else:

        d["WBC"] += bounded_uniform(1000, 5000)
        d["Neutrophils"] += bounded_uniform(5, 15)
        d["Lymphocytes"] += bounded_uniform(2, 10)

    return d


def apply_inflammatory(d):

    d["WBC"] += bounded_uniform(1200, 5000)
    d["Neutrophils"] += bounded_uniform(5, 20)
    d["Platelets"] += bounded_uniform(15000, 90000)

    return d


def apply_thrombocytopenia(d):

    d["Platelets"] -= bounded_uniform(70000, 190000)

    # Sometimes associated with mild leukopenia
    if np.random.rand() < 0.45:
        d["WBC"] -= bounded_uniform(500, 2200)

    return d


def apply_leukopenia(d):

    d["WBC"] -= bounded_uniform(1800, 4300)

    d["Neutrophils"] -= bounded_uniform(5, 18)

    if np.random.rand() < 0.55:
        d["Lymphocytes"] += bounded_uniform(3, 12)

    return d


def apply_leukocytosis(d):

    d["WBC"] += bounded_uniform(3500, 13000)

    d["Neutrophils"] += bounded_uniform(8, 25)

    return d


def apply_pancytopenia(d):

    d["Hb"] -= bounded_uniform(2, 5)

    d["RBC"] -= bounded_uniform(0.5, 1.4)

    d["WBC"] -= bounded_uniform(1800, 4500)

    d["Platelets"] -= bounded_uniform(80000, 190000)

    d["RDW"] += bounded_uniform(1, 5)

    return d


def apply_erythrocytosis(d):

    d["Hb"] += bounded_uniform(1.5, 4)

    d["RBC"] += bounded_uniform(0.7, 1.8)

    d["_Hematocrit"] += bounded_uniform(4, 10)

    return d


def apply_other_abnormal(d):

    pattern = np.random.choice(
        ["mixed", "indices", "platelet_high", "mild_multi"],
        p=[0.30, 0.25, 0.20, 0.25]
    )

    if pattern == "mixed":

        d["Hb"] -= bounded_uniform(0.5, 2)
        d["WBC"] += bounded_uniform(1000, 4000)
        d["Platelets"] -= bounded_uniform(20000, 70000)

    elif pattern == "indices":

        d["MCV"] += np.random.choice([-1, 1]) * bounded_uniform(8, 20)
        d["RDW"] += bounded_uniform(2, 5)

    elif pattern == "platelet_high":

        d["Platelets"] += bounded_uniform(60000, 150000)

    else:

        d["Hb"] -= bounded_uniform(0.5, 2)
        d["WBC"] -= bounded_uniform(500, 1800)

    return d


PRIMARY_FUNCTIONS = {
    0: apply_normal,
    1: apply_anemia,
    2: apply_infection,
    3: apply_inflammatory,
    4: apply_thrombocytopenia,
    5: apply_leukopenia,
    6: apply_leukocytosis,
    7: apply_pancytopenia,
    8: apply_erythrocytosis,
    9: apply_other_abnormal
}


# ============================================================
# DISEASE-ASSOCIATED SIGNAL MODIFIERS
# ============================================================

def apply_dengue_signal(d):

    # Prototype CBC-associated signal only.
    # Common reported CBC findings include leukopenia
    # and thrombocytopenia.
    d["Platelets"] -= bounded_uniform(70000, 190000)
    d["WBC"] -= bounded_uniform(1000, 3500)

    if np.random.rand() < 0.70:
        d["Lymphocytes"] += bounded_uniform(3, 12)

    if np.random.rand() < 0.45:
        d["_Hematocrit"] += bounded_uniform(3, 7)

    return d


def apply_malaria_signal(d):

    # Prototype malaria-associated CBC signal.
    # Keep variation broad because malaria CBC patterns overlap
    # substantially with other illnesses.
    d["Hb"] -= bounded_uniform(1.0, 4.0)

    d["Platelets"] -= bounded_uniform(40000, 140000)

    if np.random.rand() < 0.65:
        d["WBC"] -= bounded_uniform(500, 2200)

    if np.random.rand() < 0.35:
        d["Neutrophils"] += bounded_uniform(3, 10)

    return d


def apply_iron_deficiency_signal(d):

    d["Hb"] -= bounded_uniform(1.5, 4.0)
    d["MCV"] -= bounded_uniform(10, 22)
    d["MCH"] -= bounded_uniform(3, 8)
    d["MCHC"] -= bounded_uniform(1, 3)
    d["RDW"] += bounded_uniform(2, 6)
    d["RBC"] -= bounded_uniform(0.1, 0.7)

    return d


def apply_megaloblastic_signal(d):

    d["Hb"] -= bounded_uniform(1.5, 4.5)
    d["MCV"] += bounded_uniform(10, 25)
    d["MCH"] += bounded_uniform(2, 5)
    d["RDW"] += bounded_uniform(2, 7)

    # Mild reductions may occur in multiple lineages
    if np.random.rand() < 0.45:
        d["WBC"] -= bounded_uniform(500, 2200)

    if np.random.rand() < 0.40:
        d["Platelets"] -= bounded_uniform(20000, 70000)

    return d


def apply_hemolytic_signal(d):

    d["Hb"] -= bounded_uniform(1.5, 4.0)
    d["RBC"] -= bounded_uniform(0.2, 1.0)
    d["RDW"] += bounded_uniform(2, 7)

    # Reticulocytes would normally help assess hemolysis, but
    # reticulocyte count is not part of the requested CBC input.
    if np.random.rand() < 0.40:
        d["_Hematocrit"] -= bounded_uniform(3, 7)

    return d


# ============================================================
# CREATE DISEASE SIGNALS
# ============================================================

def choose_disease_signals(primary_label):

    """
    Return independent binary disease-associated signals.

    Signals are intentionally allowed to overlap.
    """

    signals = {
        "Dengue_signal": 0,
        "Malaria_signal": 0,
        "Iron_deficiency_signal": 0,
        "Megaloblastic_signal": 0,
        "Hemolytic_signal": 0
    }

    # --------------------------------------------------------
    # Baseline probabilities
    # --------------------------------------------------------

    probs = {
        "Dengue_signal": 0.035,
        "Malaria_signal": 0.025,
        "Iron_deficiency_signal": 0.080,
        "Megaloblastic_signal": 0.025,
        "Hemolytic_signal": 0.020
    }

    # --------------------------------------------------------
    # Increase probability based on primary CBC pattern
    # --------------------------------------------------------

    if primary_label in [4, 5]:
        probs["Dengue_signal"] += 0.10

    if primary_label in [1, 7]:
        probs["Iron_deficiency_signal"] += 0.12
        probs["Megaloblastic_signal"] += 0.05
        probs["Hemolytic_signal"] += 0.05

    if primary_label in [2, 4, 5]:
        probs["Malaria_signal"] += 0.07

    # --------------------------------------------------------
    # Independent sampling
    # --------------------------------------------------------

    for name, p in probs.items():
        if np.random.rand() < min(p, 0.45):
            signals[name] = 1

    return signals


# ============================================================
# HARD / EDGE CASES
# ============================================================

def apply_borderline_case(d):

    case = np.random.choice(
        ["mild_anemia", "mild_wbc", "mild_platelets", "mixed"],
        p=[0.30, 0.25, 0.25, 0.20]
    )

    if case == "mild_anemia":
        d["Hb"] -= bounded_uniform(0.5, 1.5)

    elif case == "mild_wbc":
        d["WBC"] += np.random.choice([-1, 1]) * bounded_uniform(500, 1500)

    elif case == "mild_platelets":
        d["Platelets"] += np.random.choice([-1, 1]) * bounded_uniform(
            15000, 40000
        )

    else:
        d["Hb"] -= bounded_uniform(0.3, 1)
        d["WBC"] += bounded_uniform(500, 1500)
        d["Platelets"] -= bounded_uniform(10000, 30000)

    return d


def apply_recovery_case(d):

    # Move some abnormal values toward normal.
    d["WBC"] = d["WBC"] * np.random.uniform(0.90, 1.05)
    d["Platelets"] = d["Platelets"] * np.random.uniform(0.90, 1.10)
    d["Hb"] = d["Hb"] * np.random.uniform(0.96, 1.03)

    return d


def apply_contradictory_case(d):

    # Deliberately overlapping pattern.
    d["WBC"] += bounded_uniform(1500, 4000)
    d["Platelets"] -= bounded_uniform(30000, 90000)
    d["Hb"] -= bounded_uniform(0.5, 1.5)

    return d


def apply_lab_error(d):

    # Only corrupt realistic laboratory measurements.
    lab_fields = [
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

    key = np.random.choice(lab_fields)

    d[key] *= np.random.uniform(0.85, 1.15)

    return d


# ============================================================
# SANITY / CLINICAL-STYLE CONSTRAINTS FOR SYNTHETIC DATA
# ============================================================

def clean_patient(d):

    # --------------------------------------------------------
    # Basic biological bounds
    # --------------------------------------------------------

    d["Hb"] = clamp(d["Hb"], 4.5, 21)

    d["RBC"] = clamp(d["RBC"], 1.8, 7.5)

    d["WBC"] = clamp(d["WBC"], 1200, 50000)

    d["Platelets"] = clamp(
        d["Platelets"],
        15000,
        800000
    )

    d["MCV"] = clamp(d["MCV"], 50, 125)

    d["MCH"] = clamp(d["MCH"], 15, 42)

    d["MCHC"] = clamp(d["MCHC"], 24, 39)

    d["RDW"] = clamp(d["RDW"], 9, 35)

    # --------------------------------------------------------
    # Differential percentages
    # --------------------------------------------------------

    d = normalize_differential(d)

    # --------------------------------------------------------
    # Recalculate BMI from height and weight
    # --------------------------------------------------------

    d["BMI"] = d["Weight"] / ((d["Height"] / 100) ** 2)
    d["BMI"] = clamp(d["BMI"], 12, 45)

    return d


# ============================================================
# DATASET GENERATION
# ============================================================

rows = []

# Primary class distribution.
# Keeps the dataset reasonably balanced while retaining a
# substantial normal group.
primary_distribution = [
    0.20,   # Normal
    0.13,   # Anemia
    0.12,   # Infection
    0.10,   # Inflammatory
    0.10,   # Thrombocytopenia
    0.08,   # Leukopenia
    0.08,   # Leukocytosis
    0.06,   # Pancytopenia
    0.05,   # Erythrocytosis
    0.08    # Other abnormal
]

assert np.isclose(sum(primary_distribution), 1.0)

for _ in range(TOTAL_SAMPLES):

    # --------------------------------------------------------
    # Base patient
    # --------------------------------------------------------

    patient = generate_base_person()

    # --------------------------------------------------------
    # Select primary CBC pattern
    # --------------------------------------------------------

    primary_label = np.random.choice(
        list(PRIMARY_LABELS.keys()),
        p=primary_distribution
    )

    # --------------------------------------------------------
    # Apply primary pattern
    # --------------------------------------------------------

    patient = PRIMARY_FUNCTIONS[primary_label](patient)

    # --------------------------------------------------------
    # Disease-associated signals
    # --------------------------------------------------------

    disease_signals = choose_disease_signals(primary_label)

    if disease_signals["Dengue_signal"]:
        patient = apply_dengue_signal(patient)

    if disease_signals["Malaria_signal"]:
        patient = apply_malaria_signal(patient)

    if disease_signals["Iron_deficiency_signal"]:
        patient = apply_iron_deficiency_signal(patient)

    if disease_signals["Megaloblastic_signal"]:
        patient = apply_megaloblastic_signal(patient)

    if disease_signals["Hemolytic_signal"]:
        patient = apply_hemolytic_signal(patient)

    # --------------------------------------------------------
    # Hard cases
    # --------------------------------------------------------

    r = np.random.rand()

    if r < 0.08:
        patient = apply_borderline_case(patient)

    elif r < 0.13:
        patient = apply_recovery_case(patient)

    elif r < 0.18:
        patient = apply_contradictory_case(patient)

    elif r < 0.21:
        patient = apply_lab_error(patient)

    # --------------------------------------------------------
    # Add measurement variation
    # --------------------------------------------------------

    numeric_fields = [
        "Height",
        "Weight",
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

    for field in numeric_fields:
        if np.random.rand() < 0.08:
            patient[field] = add_noise(
                patient[field],
                relative_sd=0.025
            )

    # --------------------------------------------------------
    # Clean values
    # --------------------------------------------------------

    patient = clean_patient(patient)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    # Only laboratory fields receive missing values.
    missing_candidates = [
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

    for field in missing_candidates:
        if np.random.rand() < 0.025:
            patient[field] = np.nan

    # --------------------------------------------------------
    # Remove internal generation fields
    # --------------------------------------------------------

    patient.pop("_Sex", None)
    patient.pop("_Hematocrit", None)

    # --------------------------------------------------------
    # Add primary target
    # --------------------------------------------------------

    patient["Primary_Label"] = primary_label
    patient["Primary_Label_Name"] = PRIMARY_LABELS[primary_label]

    # --------------------------------------------------------
    # Add disease signal targets
    # --------------------------------------------------------

    for signal_name, signal_value in disease_signals.items():
        patient[signal_name] = signal_value

    rows.append(patient)


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(rows)


# ============================================================
# ROUNDING
# ============================================================

ROUNDING = {
    "Age": 0,
    "Height": 1,
    "Weight": 1,
    "BMI": 1,

    "Hb": 1,
    "RBC": 2,
    "WBC": 0,
    "Platelets": 0,

    "Neutrophils": 1,
    "Lymphocytes": 1,
    "Monocytes": 1,
    "Eosinophils": 1,
    "Basophils": 1,

    "MCV": 1,
    "MCH": 1,
    "MCHC": 1,
    "RDW": 1
}

for column, decimals in ROUNDING.items():

    if column in df.columns:
        df[column] = df[column].round(decimals)


# ============================================================
# COLUMN ORDER
# ============================================================

FEATURE_COLUMNS = [
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

TARGET_COLUMNS = [
    "Primary_Label",
    "Primary_Label_Name",

    "Dengue_signal",
    "Malaria_signal",
    "Iron_deficiency_signal",
    "Megaloblastic_signal",
    "Hemolytic_signal"
]

df = df[FEATURE_COLUMNS + TARGET_COLUMNS]


# ============================================================
# SAVE
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("CBC SYNTHETIC DATASET GENERATED")
print("=" * 70)

print(f"\nTotal samples: {len(df):,}")

print("\nFeatures:")
for feature in FEATURE_COLUMNS:
    print("  -", feature)

print("\nPrimary CBC pattern distribution:")
print(
    df["Primary_Label_Name"]
    .value_counts()
    .sort_index()
)

print("\nDisease-associated signal prevalence:")
signal_columns = [
    "Dengue_signal",
    "Malaria_signal",
    "Iron_deficiency_signal",
    "Megaloblastic_signal",
    "Hemolytic_signal"
]

for signal in signal_columns:
    positive = int(df[signal].sum())
    percentage = positive / len(df) * 100

    print(
        f"  {signal:<28} "
        f"{positive:>6} "
        f"({percentage:>5.2f}%)"
    )

print("\nMissing values:")
print(df[FEATURE_COLUMNS].isnull().sum())

print("\nFirst 5 rows:")
print(df.head())

print(f"\n✅ Saved to: {OUTPUT_FILE}")