"""
Central configuration: paths, random seed and feature lists.

All paths are built relative to the project root so the project runs from any
working directory and on any machine.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_clean.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
CV_RESULTS_PATH = REPORTS_DIR / "cv_results.csv"
TUNING_RESULTS_PATH = REPORTS_DIR / "tuning_results.csv"
FEATURE_ABLATION_PATH = REPORTS_DIR / "feature_ablation.csv"
METRICS_JSON_PATH = REPORTS_DIR / "metrics.json"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"
ODDS_RATIOS_PATH = REPORTS_DIR / "logistic_odds_ratios.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
CONFUSION_MATRICES_PATH = REPORTS_DIR / "confusion_matrices.json"

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20          # 80 % train / 20 % held-out test
CV_FOLDS = 5              # stratified k-fold on the training set only

# ----------------------------------------------------------------------------
# Columns
# ----------------------------------------------------------------------------
ID_COLUMN = "customerID"   # identifier -> never used as a feature
TARGET = "Churn"           # 1 = customer left within the last month, 0 = stayed

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Internet add-on services whose raw value can be "No internet service"
INTERNET_ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Allowed values after cleaning; used to validate dashboard / API input.
ALLOWED_CATEGORIES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes"],
    "OnlineBackup": ["No", "Yes"],
    "DeviceProtection": ["No", "Yes"],
    "TechSupport": ["No", "Yes"],
    "StreamingTV": ["No", "Yes"],
    "StreamingMovies": ["No", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

# Valid numeric ranges for input validation (business-logic bounds, not
# fitted from data). tenure is in months; charges are in USD.
NUMERIC_BOUNDS = {
    "tenure": (0, 120),
    "MonthlyCharges": (0.0, 500.0),
    "TotalCharges": (0.0, 60000.0),
}
