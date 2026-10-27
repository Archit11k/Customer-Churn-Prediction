"""
Tests for data loading, cleaning, the preprocessing pipeline, the saved model
and prediction input validation.

Run:  python -m pytest -q
(The saved-model tests are skipped if `python -m src.train_model` has not been run.)
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.data_preprocessing import (
    build_preprocessor,
    clean_data,
    get_features_target,
    iqr_outliers,
    load_raw_data,
)
from src.predict import EXAMPLE_CUSTOMER, InvalidCustomerInput, predict_customers, validate_customer
from src.train_model import split_data

needs_data = pytest.mark.skipif(not config.RAW_DATA_PATH.exists(), reason="dataset not downloaded")
needs_model = pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="model not trained")


# ----------------------------------------------------------------------------
# Data loading and cleaning
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def raw():
    return load_raw_data()


@pytest.fixture(scope="module")
def clean(raw):
    return clean_data(raw)


@needs_data
def test_raw_data_shape_and_columns(raw):
    assert raw.shape == (7043, 21)
    assert {config.ID_COLUMN, config.TARGET, *config.FEATURES} <= set(raw.columns)


@needs_data
def test_clean_data_types_and_no_missing(clean):
    assert clean["TotalCharges"].dtype == float
    assert clean[config.FEATURES + [config.TARGET]].isna().sum().sum() == 0
    assert set(clean[config.TARGET].unique()) == {0, 1}


@needs_data
def test_blank_total_charges_only_for_new_customers(raw, clean):
    blank = raw["TotalCharges"].str.strip() == ""
    assert blank.sum() == 11
    assert (clean.loc[blank, "tenure"] == 0).all()
    assert (clean.loc[blank, "TotalCharges"] == 0).all()


@needs_data
def test_categories_match_allowed_values(clean):
    for col, allowed in config.ALLOWED_CATEGORIES.items():
        assert set(clean[col].unique()) <= set(allowed), col


@needs_data
def test_clean_does_not_mutate_input(raw):
    before = raw.copy()
    clean_data(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_clean_rejects_blank_total_with_positive_tenure():
    row = {c: "No" for c in config.CATEGORICAL_FEATURES}
    row.update({"customerID": "X", "SeniorCitizen": 0, "InternetService": "DSL",
                "Contract": "One year", "PaymentMethod": "Mailed check", "gender": "Male",
                "tenure": 5, "MonthlyCharges": 50.0, "TotalCharges": " ", "Churn": "No"})
    with pytest.raises(ValueError):
        clean_data(pd.DataFrame([row]))


@needs_data
def test_customer_id_excluded_from_features(clean):
    X, y = get_features_target(clean)
    assert config.ID_COLUMN not in X.columns
    assert config.TARGET not in X.columns
    assert list(X.columns) == config.FEATURES


@needs_data
def test_iqr_outliers_table(clean):
    t = iqr_outliers(clean)
    assert list(t["feature"]) == config.NUMERIC_FEATURES
    assert (t["upper_fence"] > t["lower_fence"]).all()


# ----------------------------------------------------------------------------
# Splitting and preprocessing
# ----------------------------------------------------------------------------
@needs_data
def test_split_is_stratified_reproducible_and_disjoint(clean):
    X_tr, X_te, y_tr, y_te = split_data(clean)
    X_tr2, _, _, _ = split_data(clean)
    assert X_tr.index.equals(X_tr2.index)                       # reproducible
    assert set(X_tr.index).isdisjoint(X_te.index)               # no overlap
    assert len(X_te) == round(config.TEST_SIZE * len(clean))
    assert abs(y_tr.mean() - y_te.mean()) < 0.01                # stratified


@needs_data
def test_preprocessor_output_is_numeric_and_finite(clean):
    X, _ = get_features_target(clean)
    Z = build_preprocessor().fit_transform(X)
    assert Z.shape[0] == len(X)
    assert np.isfinite(Z).all()


@needs_data
def test_small_pipeline_trains_and_predicts(clean):
    X_tr, X_te, y_tr, y_te = split_data(clean)
    pipe = Pipeline([("preprocess", build_preprocessor()),
                     ("model", LogisticRegression(max_iter=1000))]).fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert pipe.score(X_te, y_te) > y_te.value_counts(normalize=True).max() - 0.05


# ----------------------------------------------------------------------------
# Saved model and prediction
# ----------------------------------------------------------------------------
@needs_model
def test_saved_model_loads_and_uses_expected_features():
    from src.predict import load_model
    model = load_model()
    assert list(model.feature_names_in_) == config.FEATURES
    with open(config.MODEL_METADATA_PATH) as f:
        meta = json.load(f)
    assert meta["features"] == config.FEATURES


@needs_model
def test_predict_example_customer():
    out = predict_customers(EXAMPLE_CUSTOMER)
    assert len(out) == 1
    assert 0.0 <= out.loc[0, "churn_probability"] <= 1.0
    assert out.loc[0, "prediction"] in {"Churn", "No Churn"}


@needs_model
@needs_data
def test_saved_model_reproduces_reported_test_metrics(clean):
    from src.evaluate_model import compute_metrics
    from src.predict import load_model
    _, X_te, _, y_te = split_data(clean)
    with open(config.MODEL_METADATA_PATH) as f:
        meta = json.load(f)
    m = compute_metrics(load_model(), X_te, y_te)
    assert m["roc_auc"] == pytest.approx(meta["test_metrics"]["roc_auc"], abs=1e-9)
    assert m["f1"] == pytest.approx(meta["test_metrics"]["f1"], abs=1e-9)


@needs_model
def test_predict_ignores_extra_keys_and_accepts_raw_encodings():
    rec = dict(EXAMPLE_CUSTOMER, customerID="1234-ABCD", SeniorCitizen=1,
               OnlineBackup="No internet service")
    rec["InternetService"] = "No"
    for a in config.INTERNET_ADDONS:
        rec[a] = "No internet service"
    out = predict_customers(rec)
    assert len(out) == 1


def test_validation_rejects_missing_field():
    rec = dict(EXAMPLE_CUSTOMER)
    rec.pop("Contract")
    with pytest.raises(InvalidCustomerInput, match="Contract"):
        validate_customer(rec)


@pytest.mark.parametrize("field,value", [
    ("Contract", "Three year"),
    ("gender", "Unknown"),
    ("tenure", -1),
    ("tenure", "abc"),
    ("MonthlyCharges", 10_000),
])
def test_validation_rejects_bad_values(field, value):
    with pytest.raises(InvalidCustomerInput):
        validate_customer(dict(EXAMPLE_CUSTOMER, **{field: value}))


def test_validation_rejects_inconsistent_services():
    rec = dict(EXAMPLE_CUSTOMER, InternetService="No", OnlineSecurity="Yes")
    with pytest.raises(InvalidCustomerInput, match="internet"):
        validate_customer(rec)
    rec = dict(EXAMPLE_CUSTOMER, PhoneService="No", MultipleLines="Yes")
    with pytest.raises(InvalidCustomerInput, match="PhoneService"):
        validate_customer(rec)
