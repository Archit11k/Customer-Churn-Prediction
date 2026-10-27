"""
Evaluation utilities: metrics, confusion matrices, ROC / precision-recall
curves, Logistic Regression odds ratios, permutation feature importance and a
validation curve that illustrates under- and overfitting.

``generate_evaluation_artifacts`` is called by ``train_model`` after training.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import validation_curve

from src import config

sns.set_theme(style="whitegrid", context="notebook")


def compute_metrics(pipe, X, y, threshold: float = 0.5) -> dict:
    """Accuracy, precision, recall, F1, ROC-AUC and confusion matrix on (X, y)."""
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(fitted, X_test, y_test, stage="tuned"):
    items = [(n, p) for (n, s), p in fitted.items() if s == stage]
    fig, axes = plt.subplots(1, len(items), figsize=(4.4 * len(items), 4))
    for ax, (name, pipe) in zip(np.atleast_1d(axes), items):
        cm = confusion_matrix(y_test, pipe.predict(X_test), labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues", ax=ax,
                    xticklabels=["No churn", "Churn"], yticklabels=["No churn", "Churn"])
        ax.set(title=name, xlabel="Predicted", ylabel="Actual")
    fig.suptitle(f"Confusion matrices on the held-out test set ({stage} models)", y=1.03)
    _save(fig, f"model_confusion_matrices_{stage}.png")


def plot_curves(fitted, X_test, y_test):
    items = [(n, p) for (n, s), p in fitted.items() if s == "tuned"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, pipe in items:
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    axes[0].set_title("ROC curves (test set)")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[1].axhline(y_test.mean(), ls="--", c="grey", lw=1, label="Churn base rate")
    axes[1].set_title("Precision-recall curves (test set)")
    axes[1].legend(loc="upper right", fontsize=9)
    _save(fig, "model_roc_pr_curves.png")


def plot_metric_comparison(comparison: pd.DataFrame):
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
    d = comparison.copy()
    d["label"] = d["model"] + " (" + d["stage"] + ")"
    long = d.melt(id_vars="label", value_vars=metrics, var_name="metric", value_name="score")
    long["metric"] = long["metric"].str.replace("test_", "")
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(data=long, x="metric", y="score", hue="label", ax=ax)
    ax.set(ylim=(0, 1), title="Test-set metrics: baseline vs tuned models", ylabel="Score", xlabel="")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    _save(fig, "model_metric_comparison.png")


def logistic_odds_ratios(lr_pipe, X_train) -> pd.DataFrame:
    """Coefficients and odds ratios of a fitted Logistic Regression pipeline."""
    pre, model = lr_pipe.named_steps["preprocess"], lr_pipe.named_steps["model"]
    names = pre.get_feature_names_out()
    coef = model.coef_.ravel()
    df = pd.DataFrame({"feature": names, "coefficient": coef, "odds_ratio": np.exp(coef)})
    # For standardised numeric features one unit = one standard deviation.
    sds = X_train[config.NUMERIC_FEATURES].std()
    df["unit"] = df["feature"].map(lambda f: f"+1 SD ({sds[f]:.1f})" if f in sds else "vs reference")
    return df.reindex(df["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def plot_odds_ratios(odds: pd.DataFrame):
    d = odds[odds["coefficient"] != 0].sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(d) + 1.5))
    colors = ["#DD8452" if c > 0 else "#4C72B0" for c in d["coefficient"]]
    ax.barh(d["feature"], d["odds_ratio"], color=colors)
    ax.axvline(1, c="k", lw=1)
    ax.set_xscale("log")
    ax.set(title="Logistic Regression odds ratios (log scale)\n>1 = higher churn odds, <1 = lower",
           xlabel="Odds ratio")
    _save(fig, "model_logistic_odds_ratios.png")


def permutation_feature_importance(pipe, X_test, y_test) -> pd.DataFrame:
    """Drop in ROC-AUC when each raw feature is shuffled (model-agnostic)."""
    r = permutation_importance(pipe, X_test, y_test, scoring="roc_auc", n_repeats=10,
                               random_state=config.RANDOM_STATE, n_jobs=-1)
    return (pd.DataFrame({"feature": X_test.columns, "importance_mean": r.importances_mean,
                          "importance_std": r.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))


def plot_feature_importance(imp: pd.DataFrame, model_name: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    d = imp.sort_values("importance_mean")
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color="#55A868")
    ax.set(title=f"Permutation importance - {model_name}\n(drop in test ROC-AUC when feature is shuffled)",
           xlabel="Mean decrease in ROC-AUC")
    _save(fig, "model_permutation_importance.png")


def plot_validation_curve(X_train, y_train):
    """Decision-tree depth vs train/CV ROC-AUC: shows under- and overfitting."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    from src.data_preprocessing import build_preprocessor

    pipe = Pipeline([("preprocess", build_preprocessor()),
                     ("model", DecisionTreeClassifier(class_weight="balanced",
                                                      random_state=config.RANDOM_STATE))])
    depths = np.arange(1, 21)
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    train_s, val_s = validation_curve(pipe, X_train, y_train, param_name="model__max_depth",
                                      param_range=depths, cv=cv, scoring="roc_auc", n_jobs=-1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(depths, train_s.mean(1), "o-", label="Training ROC-AUC")
    ax.plot(depths, val_s.mean(1), "o-", label="Cross-validation ROC-AUC")
    ax.fill_between(depths, val_s.mean(1) - val_s.std(1), val_s.mean(1) + val_s.std(1), alpha=0.2)
    best = depths[val_s.mean(1).argmax()]
    ax.axvline(best, ls="--", c="grey", label=f"best CV depth = {best}")
    ax.set(xlabel="max_depth", ylabel="ROC-AUC",
           title="Validation curve - Decision Tree (training data, 5-fold CV)\n"
                 "left: underfitting (both low)   right: overfitting (gap widens)")
    ax.legend()
    _save(fig, "model_validation_curve_tree_depth.png")
    return pd.DataFrame({"max_depth": depths, "train_roc_auc": train_s.mean(1),
                         "cv_roc_auc": val_s.mean(1)})


def generate_evaluation_artifacts(results: dict) -> None:
    fitted = results["fitted"]
    X_train, y_train = results["X_train"], results["y_train"]
    X_test, y_test = results["X_test"], results["y_test"]
    selected = results["selected_name"]

    plot_confusion_matrices(fitted, X_test, y_test, "baseline")
    plot_confusion_matrices(fitted, X_test, y_test, "tuned")
    plot_curves(fitted, X_test, y_test)
    plot_metric_comparison(results["comparison"])

    odds = logistic_odds_ratios(fitted[("Logistic Regression", "tuned")], X_train)
    odds.to_csv(config.ODDS_RATIOS_PATH, index=False)
    plot_odds_ratios(odds)

    imp = permutation_feature_importance(fitted[(selected, "tuned")], X_test, y_test)
    imp.to_csv(config.FEATURE_IMPORTANCE_PATH, index=False)
    plot_feature_importance(imp, selected)

    vc = plot_validation_curve(X_train, y_train)
    vc.to_csv(config.REPORTS_DIR / "validation_curve_tree_depth.csv", index=False)
