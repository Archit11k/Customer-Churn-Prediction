"""
Data loading, cleaning and preprocessing for the Telco Customer Churn project.

Two kinds of preprocessing are deliberately kept separate:

1. ``clean_data`` - deterministic, row-by-row fixes (type conversion, fixing
   inconsistent labels). These use no statistics learned from the data, so
   applying them before the train/test split cannot leak information.

2. ``build_preprocessor`` - a scikit-learn ``ColumnTransformer`` (scaling,
   one-hot encoding). It *learns* parameters (means, std-devs, categories), so
   it lives inside the model ``Pipeline`` and is fitted on training folds only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # works both as ``python -m src.x`` and when src/ is on sys.path
    from src import config
except ImportError:  # pragma: no cover
    import config  # type: ignore


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_raw_data(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw IBM Telco CSV, with a helpful error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download 'WA_Fn-UseC_-Telco-Customer-Churn.csv' from Kaggle "
            "(https://www.kaggle.com/datasets/blastchar/telco-customer-churn) "
            "or 'Telco-Customer-Churn.csv' from "
            "https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data "
            "and save it as data/raw/Telco-Customer-Churn.csv"
        )
    df = pd.read_csv(path)
    expected = set(config.FEATURES + [config.ID_COLUMN, config.TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df


# ----------------------------------------------------------------------------
# Data-quality checks (used by the EDA notebook and report)
# ----------------------------------------------------------------------------
def data_quality_report(df: pd.DataFrame) -> dict:
    """Summarise missing values, blanks, duplicates and dtypes of the raw data."""
    blank_strings = {
        col: int(df[col].astype(str).str.strip().eq("").sum())
        for col in df.columns
        if df[col].dtype == object
    }
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "missing_values (NaN)": {k: int(v) for k, v in df.isna().sum().items() if v},
        "blank_strings": {k: v for k, v in blank_strings.items() if v},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rows_ignoring_id": int(df.drop(columns=[config.ID_COLUMN]).duplicated().sum()),
        "duplicate_customer_ids": int(df[config.ID_COLUMN].duplicated().sum()),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
    }


def iqr_outliers(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Return Q1, Q3, IQR, Tukey fences and outlier count for numeric columns."""
    columns = columns or config.NUMERIC_FEATURES
    rows = []
    for col in columns:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lower) | (s > upper)).sum())
        rows.append({
            "feature": col, "Q1": q1, "Q3": q3, "IQR": iqr,
            "lower_fence": lower, "upper_fence": upper,
            "min": s.min(), "max": s.max(), "n_outliers": n_out,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic cleaning rules (no statistics are learned here).

    1. TotalCharges is stored as text; 11 rows contain a blank string ' '.
       All of these customers have tenure == 0 (they joined in the current
       month and have not been billed yet), so the blank is replaced by 0.
    2. SeniorCitizen is coded 0/1 while every other yes/no column uses the
       strings 'No'/'Yes'; it is recoded to 'No'/'Yes' for consistency.
    3. 'No internet service' / 'No phone service' duplicate information that
       is already in InternetService / PhoneService. They are mapped to 'No'
       so that the add-on columns mean "has this add-on: yes/no" and the
       one-hot columns are not perfectly collinear.
    4. The target Churn is mapped to 1 (Yes) / 0 (No).
    5. Exact duplicate rows (if any) are dropped.
    """
    df = df.copy()

    # 1. TotalCharges: text -> number; blank only occurs for brand-new customers
    total = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    blank_mask = total.isna()
    if blank_mask.any():
        # Sanity check of the assumption documented above.
        if not (df.loc[blank_mask, "tenure"] == 0).all():
            raise ValueError("Blank TotalCharges found for customers with tenure > 0")
        total = total.fillna(0.0)
    df["TotalCharges"] = total.astype(float)

    # 2. Consistent yes/no coding
    if df["SeniorCitizen"].dtype != object:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # 3. Collapse redundant "No ... service" labels
    for col in config.INTERNET_ADDONS:
        df[col] = df[col].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    # Strip stray whitespace in all text columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # 4. Target encoding
    if df[config.TARGET].dtype == object:
        df[config.TARGET] = df[config.TARGET].map({"No": 0, "Yes": 1})
    if df[config.TARGET].isna().any():
        raise ValueError("Unexpected value in the Churn column")
    df[config.TARGET] = df[config.TARGET].astype(int)

    # 5. Duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def load_clean_data(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Convenience wrapper: load the raw CSV and clean it."""
    return clean_data(load_raw_data(path))


def save_processed(df: pd.DataFrame, path=config.PROCESSED_DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def get_features_target(df: pd.DataFrame):
    """
    Split a cleaned frame into X (model features) and y (target).

    customerID is excluded: it is a unique random label, carries no
    behavioural information, and a model could only memorise it (overfit).
    """
    X = df[config.FEATURES].copy()
    y = df[config.TARGET].copy()
    return X, y


# ----------------------------------------------------------------------------
# Preprocessing pipeline
# ----------------------------------------------------------------------------
def build_preprocessor(numeric_features=None, categorical_features=None) -> ColumnTransformer:
    """
    ColumnTransformer used inside every model pipeline.

    - Numeric: median imputation (safety net for missing dashboard input) and
      standardisation (needed for Logistic Regression / LDA; harmless for trees).
    - Categorical: most-frequent imputation and one-hot encoding with the first
      category dropped, so Logistic Regression coefficients read as
      "compared with the reference category".
    """
    numeric_features = numeric_features if numeric_features is not None else config.NUMERIC_FEATURES
    categorical_features = (
        categorical_features if categorical_features is not None else config.CATEGORICAL_FEATURES
    )

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", numeric_pipe, list(numeric_features)),
            ("cat", categorical_pipe, list(categorical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def tenure_group(tenure: pd.Series) -> pd.Series:
    """Bucket tenure (months) into readable bands - used for EDA only."""
    bins = [-1, 12, 24, 48, 72]
    labels = ["0-12 m", "13-24 m", "25-48 m", "49-72 m"]
    return pd.cut(tenure, bins=bins, labels=labels)


def count_addons(df: pd.DataFrame) -> pd.Series:
    """Number of internet add-on services a customer subscribes to (0-6)."""
    return (df[config.INTERNET_ADDONS] == "Yes").sum(axis=1)


if __name__ == "__main__":
    raw = load_raw_data()
    print(data_quality_report(raw))
    clean = clean_data(raw)
    save_processed(clean)
    print(f"Saved cleaned data: {clean.shape} -> {config.PROCESSED_DATA_PATH}")
    print(np.round(clean[config.TARGET].mean(), 4), "churn rate")
