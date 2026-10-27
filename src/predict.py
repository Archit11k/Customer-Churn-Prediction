"""
Load the saved churn pipeline and predict for new customers.

The saved object is a full scikit-learn Pipeline (ColumnTransformer + model),
so raw, cleaned customer attributes go in and no manual encoding is needed.

Example:
    python -m src.predict
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from src import config


class InvalidCustomerInput(ValueError):
    """Raised when customer attributes are missing, unknown or out of range."""


@lru_cache(maxsize=1)
def load_model(path=config.MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m src.train_model` first."
        )
    return joblib.load(path)


def validate_customer(record: dict) -> dict:
    """
    Check one customer's attributes and return a cleaned copy.

    - every model feature must be present (extra keys such as customerID are ignored)
    - categorical values must be one of the allowed categories
    - numeric values must be numbers within sensible business bounds
    - add-on services cannot be 'Yes' without the underlying service
    """
    missing = [f for f in config.FEATURES if f not in record or record[f] is None]
    if missing:
        raise InvalidCustomerInput(f"Missing values for: {missing}")

    clean = {}
    for col in config.CATEGORICAL_FEATURES:
        value = record[col]
        # Accept the raw-dataset encodings too, and map them like clean_data does.
        if col == "SeniorCitizen" and value in (0, 1, "0", "1"):
            value = "Yes" if int(value) == 1 else "No"
        if value in ("No internet service", "No phone service"):
            value = "No"
        if value not in config.ALLOWED_CATEGORIES[col]:
            raise InvalidCustomerInput(
                f"{col}={value!r} is not allowed; expected one of {config.ALLOWED_CATEGORIES[col]}"
            )
        clean[col] = value

    for col in config.NUMERIC_FEATURES:
        try:
            value = float(record[col])
        except (TypeError, ValueError):
            raise InvalidCustomerInput(f"{col} must be a number, got {record[col]!r}") from None
        lo, hi = config.NUMERIC_BOUNDS[col]
        if not lo <= value <= hi:
            raise InvalidCustomerInput(f"{col}={value} is outside the allowed range [{lo}, {hi}]")
        clean[col] = value

    # Logical consistency between services (mirrors the structure of the data)
    if clean["InternetService"] == "No":
        bad = [c for c in config.INTERNET_ADDONS if clean[c] == "Yes"]
        if bad:
            raise InvalidCustomerInput(f"{bad} require an internet service")
    if clean["PhoneService"] == "No" and clean["MultipleLines"] == "Yes":
        raise InvalidCustomerInput("MultipleLines requires PhoneService")

    return clean


def predict_customers(records, threshold: float = 0.5) -> pd.DataFrame:
    """
    Predict churn for one or more customers.

    Returns a DataFrame with churn_probability and prediction ('Churn' /
    'No Churn'). The probability is the model's estimate, not a certainty.
    """
    if isinstance(records, dict):
        records = [records]
    cleaned = [validate_customer(r) for r in records]
    X = pd.DataFrame(cleaned)[config.FEATURES]   # exact column order used in training
    model = load_model()
    proba = model.predict_proba(X)[:, 1]
    return pd.DataFrame({
        "churn_probability": proba,
        "prediction": ["Churn" if p >= threshold else "No Churn" for p in proba],
    })


EXAMPLE_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
    "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 255.0,
}


if __name__ == "__main__":
    print(predict_customers(EXAMPLE_CUSTOMER).to_string(index=False))
