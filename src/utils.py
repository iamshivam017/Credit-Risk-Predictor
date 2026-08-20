"""
src/utils.py
-------------
Portable path definitions and shared constants for the Credit Risk
Predictor project. Every other module imports paths from here instead
of hardcoding strings, so the whole project can be moved/cloned
anywhere and still work.
"""

from pathlib import Path

# ── Portable project paths ───────────────────────────────────────────────────
# Path(__file__) resolves relative to THIS file's location on disk, not the
# current working directory a script happens to be launched from — so
# `streamlit run app.py`, `python train.py`, and running notebook cells from
# a different folder all resolve the same paths correctly.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR       = PROJECT_ROOT / "data"
RAW_DATA_DIR   = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
MODELS_DIR     = PROJECT_ROOT / "models"
PLOTS_DIR      = PROJECT_ROOT / "plots"

INTERNAL_PATH   = RAW_DATA_DIR / "internal_bank_dataset.xlsx"
EXTERNAL_PATH   = RAW_DATA_DIR / "external_cibil_dataset.xlsx"
UNSEEN_PATH     = RAW_DATA_DIR / "unseen_dataset.xlsx"
FEATURE_DESC_PATH = RAW_DATA_DIR / "features_target_description.xlsx"

MERGED_CACHE_PATH = PROCESSED_DIR / "merged_clean.pkl"

MODEL_DEPLOYMENT_PATH = MODELS_DIR / "model_deployment.pkl"
MODEL_FULL_PATH       = MODELS_DIR / "model_full_with_credit_score.pkl"
METADATA_PATH         = MODELS_DIR / "metadata.json"

for _dir in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR, PLOTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ── Shared constants (single source of truth) ───────────────────────────────
# These must stay IDENTICAL across cleaning, feature engineering, training,
# and prediction — defining them once here (instead of copy-pasting into
# every script/notebook cell) is what prevents "train/serve skew" bugs.

RANDOM_STATE = 42

SENTINEL_VALUE = -99999
SENTINEL_COLUMN_DROP_THRESHOLD = 10_000  # drop external column if > this many sentinel rows

# Ordinal mapping for EDUCATION — order matters, values must stay in sync
# with train_model.py and dashboard_utils.py's manual-entry form.
EDUCATION_ORDER = {
    "SSC": 1,
    "12TH": 2,
    "UNDER GRADUATE": 3,
    "GRADUATE": 4,
    "PROFESSIONAL": 4,
    "POST-GRADUATE": 5,
    "OTHERS": 1,
}

# Categorical columns one-hot encoded (order matters for reproducible dummy columns)
ONEHOT_COLUMNS = ["MARITALSTATUS", "GENDER", "last_prod_enq2", "first_prod_enq2"]

# The 42 columns present in unseen_dataset.xlsx (before one-hot expansion).
# This is the ground truth for what "deployment-safe" features look like —
# computed once via `sorted(pd.read_excel(UNSEEN_PATH).columns)` and pinned
# here so every module agrees on it without needing to re-read the file.
UNSEEN_BASE_COLUMNS = [
    "Age_Newest_TL", "Age_Oldest_TL", "CC_Flag", "CC_TL", "CC_enq_L12m",
    "EDUCATION", "GENDER", "GL_Flag", "HL_Flag", "Home_TL", "MARITALSTATUS",
    "NETMONTHLYINCOME", "Other_TL", "PL_Flag", "PL_TL", "PL_enq_L12m",
    "Secured_TL", "Time_With_Curr_Empr", "Tot_Missed_Pmnt", "Tot_TL_closed_L12M",
    "Unsecured_TL", "enq_L3m", "first_prod_enq2", "last_prod_enq2",
    "max_recent_level_of_deliq", "num_dbt", "num_dbt_12mts", "num_deliq_6_12mts",
    "num_lss", "num_std_12mts", "num_sub", "num_sub_12mts", "num_sub_6mts",
    "num_times_60p_dpd", "pct_CC_enq_L6m_of_ever", "pct_PL_enq_L6m_of_ever",
    "pct_tl_closed_L12M", "pct_tl_closed_L6M", "pct_tl_open_L6M",
    "recent_level_of_deliq", "time_since_recent_enq", "time_since_recent_payment",
]

TARGET_CLASSES = ["P1", "P2", "P3", "P4"]

TARGET_CLASS_MEANING = {
    "P1": "Lowest risk — auto-approve",
    "P2": "Low\u2013moderate risk",
    "P3": "Moderate\u2013high risk",
    "P4": "Highest risk — likely reject",
}
