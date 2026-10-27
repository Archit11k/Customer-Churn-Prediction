"""
Model training, cross-validation, hyperparameter tuning and model selection.

Workflow (the held-out test set is touched only in step 6):

1. Clean data, drop customerID, stratified 80/20 train-test split (seed 42).
2. Baseline: 5-fold stratified CV on the TRAINING set for a majority-class
   dummy model, Logistic Regression, Decision Tree, Random Forest and LDA.
3. Feature-set ablation (Logistic Regression, training CV only).
4. Hyperparameter tuning with GridSearchCV / RandomizedSearchCV, again with
   5-fold CV on the training set only (scoring = ROC-AUC).
5. Model selection: the tuned model with the highest mean CV ROC-AUC.
6. Every model is refitted on the full training set and scored once on the
   test set; the test numbers are reported, never used to choose anything.
7. The selected pipeline (preprocessing + model) is saved with joblib.

Run:  python -m src.train_model
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.metrics import make_scorer, precision_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.data_preprocessing import (
    build_preprocessor,
    count_addons,
    get_features_target,
    load_clean_data,
    save_processed,
)
from src.evaluate_model import compute_metrics

SCORING = {
    "accuracy": "accuracy",
    # zero_division=0: the majority-class baseline never predicts churn, so
    # precision is undefined for it; report 0 instead of raising a warning.
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def make_pipeline(estimator, numeric=None, categorical=None) -> Pipeline:
    """Preprocessing + estimator, so scaling/encoding is fitted per CV fold."""
    return Pipeline([
        ("preprocess", build_preprocessor(numeric, categorical)),
        ("model", estimator),
    ])


def baseline_models() -> dict:
    """
    Untuned models with sensible defaults.

    Class imbalance (about 26.5 % churners) is handled with
    class_weight='balanced' (LR, DT, RF) and equal priors (LDA), which makes a
    missed churner cost as much in total as the more numerous non-churners.
    """
    rs = config.RANDOM_STATE
    return {
        "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=rs),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=rs),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=rs, n_jobs=-1),
        "LDA": LinearDiscriminantAnalysis(priors=[0.5, 0.5]),
    }


def search_spaces() -> dict:
    """Hyperparameter search for each real model (dummy model is not tuned)."""
    rs = config.RANDOM_STATE
    return {
        "Logistic Regression": GridSearchCV(
            make_pipeline(LogisticRegression(max_iter=5000, class_weight="balanced",
                                             solver="liblinear", random_state=rs)),
            param_grid={
                "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
                "model__penalty": ["l1", "l2"],
            },
            scoring=SCORING, refit="roc_auc", cv=cv_splitter(), n_jobs=-1,
            return_train_score=True,
        ),
        "Decision Tree": GridSearchCV(
            make_pipeline(DecisionTreeClassifier(class_weight="balanced", random_state=rs)),
            param_grid={
                "model__max_depth": [3, 4, 5, 6, 8, 10, None],
                "model__min_samples_leaf": [1, 10, 25, 50, 100],
                "model__criterion": ["gini", "entropy"],
            },
            scoring=SCORING, refit="roc_auc", cv=cv_splitter(), n_jobs=-1,
            return_train_score=True,
        ),
        "Random Forest": RandomizedSearchCV(
            make_pipeline(RandomForestClassifier(class_weight="balanced", random_state=rs, n_jobs=1)),
            param_distributions={
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [4, 6, 8, 10, 12, None],
                "model__min_samples_leaf": [1, 5, 10, 20, 40],
                "model__max_features": ["sqrt", "log2", 0.5],
            },
            n_iter=30, scoring=SCORING, refit="roc_auc", cv=cv_splitter(), n_jobs=-1,
            random_state=rs, return_train_score=True,
        ),
        "LDA": GridSearchCV(
            make_pipeline(LinearDiscriminantAnalysis(priors=[0.5, 0.5])),
            param_grid=[
                {"model__solver": ["svd"]},
                {"model__solver": ["lsqr"], "model__shrinkage": [None, "auto", 0.01, 0.1, 0.3, 0.5]},
            ],
            scoring=SCORING, refit="roc_auc", cv=cv_splitter(), n_jobs=-1,
            return_train_score=True,
        ),
    }


# ----------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------
def split_data(df: pd.DataFrame):
    X, y = get_features_target(df)
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y,
                            random_state=config.RANDOM_STATE)


def summarise_cv(cv_res: dict) -> dict:
    out = {}
    for m in SCORING:
        out[f"cv_{m}_mean"] = float(np.mean(cv_res[f"test_{m}"]))
        out[f"cv_{m}_std"] = float(np.std(cv_res[f"test_{m}"]))
        out[f"train_{m}_mean"] = float(np.mean(cv_res[f"train_{m}"]))
    return out


def run_baseline_cv(X_train, y_train) -> pd.DataFrame:
    rows = []
    for name, est in baseline_models().items():
        res = cross_validate(make_pipeline(est), X_train, y_train, cv=cv_splitter(),
                             scoring=SCORING, return_train_score=True, n_jobs=-1)
        rows.append({"model": name, "stage": "baseline", **summarise_cv(res)})
    return pd.DataFrame(rows)


def run_feature_ablation(X_train, y_train) -> pd.DataFrame:
    """
    Compare candidate feature sets with Logistic Regression (training CV only).

    A: all 19 cleaned features
    B: A without TotalCharges (almost exactly tenure x MonthlyCharges)
    C: A plus NumAddOns (count of internet add-on services)
    D: A without gender and PhoneService (no significant chi-square association)
    """
    Xc = X_train.copy()
    Xc["NumAddOns"] = count_addons(Xc)
    num, cat = config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES
    sets = {
        "A: all 19 features": (num, cat),
        "B: without TotalCharges": ([c for c in num if c != "TotalCharges"], cat),
        "C: + NumAddOns (derived)": (num + ["NumAddOns"], cat),
        "D: without gender & PhoneService": (num, [c for c in cat if c not in ("gender", "PhoneService")]),
    }
    rows = []
    for label, (n, c) in sets.items():
        est = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE)
        res = cross_validate(make_pipeline(est, n, c), Xc, y_train, cv=cv_splitter(),
                             scoring={"roc_auc": "roc_auc", "f1": "f1"}, n_jobs=-1)
        rows.append({
            "feature_set": label, "n_features": len(n) + len(c),
            "cv_roc_auc_mean": res["test_roc_auc"].mean(), "cv_roc_auc_std": res["test_roc_auc"].std(),
            "cv_f1_mean": res["test_f1"].mean(), "cv_f1_std": res["test_f1"].std(),
        })
    return pd.DataFrame(rows)


def run_tuning(X_train, y_train):
    tuned, rows, all_results = {}, [], []
    for name, search in search_spaces().items():
        t0 = time.time()
        search.fit(X_train, y_train)
        i = search.best_index_
        cvr = search.cv_results_
        row = {"model": name, "stage": "tuned",
               "best_params": json.dumps({k.replace("model__", ""): v for k, v in search.best_params_.items()},
                                         default=str)}
        for m in SCORING:
            row[f"cv_{m}_mean"] = float(cvr[f"mean_test_{m}"][i])
            row[f"cv_{m}_std"] = float(cvr[f"std_test_{m}"][i])
            row[f"train_{m}_mean"] = float(cvr[f"mean_train_{m}"][i])
        row["n_candidates"] = len(cvr["params"])
        row["search_seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        tuned[name] = search.best_estimator_
        cv_df = pd.DataFrame(cvr)[["params", "mean_test_roc_auc", "std_test_roc_auc",
                                   "mean_train_roc_auc", "mean_test_f1", "rank_test_roc_auc"]]
        cv_df.insert(0, "model", name)
        all_results.append(cv_df)
        print(f"  tuned {name:<20} CV ROC-AUC={row['cv_roc_auc_mean']:.4f}  ({row['search_seconds']} s)")
    return tuned, pd.DataFrame(rows), pd.concat(all_results, ignore_index=True)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run_training(verbose: bool = True) -> dict:
    df = load_clean_data()
    save_processed(df)
    X_train, X_test, y_train, y_test = split_data(df)
    if verbose:
        print(f"Train: {X_train.shape}, churn={y_train.mean():.3f} | "
              f"Test: {X_test.shape}, churn={y_test.mean():.3f}")

    # 2. Baseline cross-validation
    baseline_cv = run_baseline_cv(X_train, y_train)
    if verbose:
        print(baseline_cv[["model", "cv_accuracy_mean", "cv_recall_mean", "cv_f1_mean", "cv_roc_auc_mean"]]
              .round(4).to_string(index=False))

    # 3. Feature-set ablation
    ablation = run_feature_ablation(X_train, y_train)
    ablation.to_csv(config.FEATURE_ABLATION_PATH, index=False)
    if verbose:
        print(ablation.round(4).to_string(index=False))

    # 4. Hyperparameter tuning
    tuned, tuned_cv, all_cv = run_tuning(X_train, y_train)
    all_cv.to_csv(config.TUNING_RESULTS_PATH, index=False)

    # 5. Model selection - CV ROC-AUC on training data only
    selected_name = tuned_cv.sort_values("cv_roc_auc_mean", ascending=False).iloc[0]["model"]
    if verbose:
        print(f"Selected model (highest mean CV ROC-AUC): {selected_name}")

    # 6. Fit baselines on the full training set; score everything once on test
    fitted = {}
    for name, est in baseline_models().items():
        fitted[(name, "baseline")] = make_pipeline(est).fit(X_train, y_train)
    for name, pipe in tuned.items():
        fitted[(name, "tuned")] = pipe  # best_estimator_ is already refitted on all training data

    cv_table = pd.concat([baseline_cv, tuned_cv], ignore_index=True)
    rows, confusion = [], {}
    for (name, stage), pipe in fitted.items():
        test_m = compute_metrics(pipe, X_test, y_test)
        cm = test_m.pop("confusion_matrix")
        confusion[f"{name} ({stage})"] = cm
        cv_row = cv_table[(cv_table.model == name) & (cv_table.stage == stage)].iloc[0].to_dict()
        rows.append({
            "model": name, "stage": stage,
            "cv_roc_auc_mean": cv_row["cv_roc_auc_mean"], "cv_roc_auc_std": cv_row["cv_roc_auc_std"],
            "cv_f1_mean": cv_row["cv_f1_mean"], "cv_recall_mean": cv_row["cv_recall_mean"],
            "cv_accuracy_mean": cv_row["cv_accuracy_mean"],
            "train_roc_auc_mean": cv_row["train_roc_auc_mean"],
            **{f"test_{k}": v for k, v in test_m.items()},
            "best_params": cv_row.get("best_params", ""),
            "selected": bool(name == selected_name and stage == "tuned"),
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(config.MODEL_COMPARISON_PATH, index=False)
    cv_table.to_csv(config.CV_RESULTS_PATH, index=False)
    with open(config.CONFUSION_MATRICES_PATH, "w") as f:
        json.dump(confusion, f, indent=2)

    # 7. Persist the selected pipeline + metadata
    selected = tuned[selected_name]
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, config.MODEL_PATH)
    sel_row = comparison[comparison.selected].iloc[0]
    metadata = {
        "model_name": selected_name,
        "selection_rule": "highest mean 5-fold CV ROC-AUC on the training set (tuned models)",
        "best_params": json.loads(sel_row["best_params"]),
        "decision_threshold": 0.5,
        "features": config.FEATURES,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "allowed_categories": config.ALLOWED_CATEGORIES,
        "numeric_bounds": config.NUMERIC_BOUNDS,
        "cv_roc_auc_mean": float(sel_row["cv_roc_auc_mean"]),
        "test_metrics": {k.replace("test_", ""): float(sel_row[k]) for k in sel_row.index if k.startswith("test_")},
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "random_state": config.RANDOM_STATE,
        "dataset_file": config.RAW_DATA_PATH.name,
        "dataset_sha256": file_sha256(config.RAW_DATA_PATH),
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(config.MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "fitted": fitted, "selected_name": selected_name, "comparison": comparison,
        "ablation": ablation, "cv_table": cv_table,
    }


if __name__ == "__main__":
    from src.evaluate_model import generate_evaluation_artifacts

    results = run_training()
    generate_evaluation_artifacts(results)
    cols = ["model", "stage", "cv_roc_auc_mean", "test_accuracy", "test_precision",
            "test_recall", "test_f1", "test_roc_auc", "selected"]
    print(results["comparison"][cols].round(4).to_string(index=False))
    print(f"Saved model -> {config.MODEL_PATH}")
