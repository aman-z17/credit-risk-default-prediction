"""Tune, train, and evaluate a random-forest credit-risk model."""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)


DATA_PATH = PROJECT_ROOT / "data" / "credit_risk_data.csv"
IMAGE_DIR = PROJECT_ROOT / "images"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"
TARGET = "default"
RANDOM_STATE = 42

BLUE = "#3B82B8"
ORANGE = "#E07A3F"


def split_data(data: pd.DataFrame):
    features = data.drop(columns=TARGET)
    target = data[TARGET]
    return train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def tune_model(x_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    forest = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    parameter_space = {
        "n_estimators": [100, 180, 250],
        "max_depth": [6, 10, 14, None],
        "min_samples_split": [2, 10, 25],
        "min_samples_leaf": [1, 3, 8, 15],
        "max_features": ["sqrt", 0.6, 1.0],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }
    cross_validation = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    search = RandomizedSearchCV(
        estimator=forest,
        param_distributions=parameter_space,
        n_iter=6,
        scoring="roc_auc",
        cv=cross_validation,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        verbose=1,
        return_train_score=True,
    )
    search.fit(x_train, y_train)
    return search


def evaluate(y_true: pd.Series, predictions, probabilities) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
    }


def save_confusion_matrix(y_true: pd.Series, predictions) -> None:
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["No default", "Default"],
        cmap="Oranges",
        colorbar=False,
    )
    display.ax_.set_title("Random Forest Confusion Matrix", weight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(
        IMAGE_DIR / "random_forest_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def save_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(RESULT_DIR / "random_forest_feature_importance.csv", index=False)

    chart_data = importance.sort_values("importance")
    plt.figure(figsize=(8, 5.5))
    plt.barh(chart_data["feature"], chart_data["importance"], color=ORANGE)
    plt.title("Random Forest Feature Importance", weight="bold", pad=14)
    plt.xlabel("Mean decrease in impurity")
    plt.ylabel("")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        IMAGE_DIR / "random_forest_feature_importance.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
    return importance


def save_model_comparison(
    y_test: pd.Series,
    forest_probabilities,
    forest_metrics: dict[str, float],
    x_test: pd.DataFrame,
) -> None:
    logistic_model = joblib.load(MODEL_DIR / "logistic_regression.joblib")
    logistic_probabilities = logistic_model.predict_proba(x_test)[:, 1]

    figure, axis = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(
        y_test,
        logistic_probabilities,
        name="Logistic regression",
        ax=axis,
        curve_kwargs={"color": BLUE},
    )
    RocCurveDisplay.from_predictions(
        y_test,
        forest_probabilities,
        name="Random forest",
        ax=axis,
        curve_kwargs={"color": ORANGE},
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    axis.set_title("Held-Out Test ROC Curves", weight="bold", pad=14)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(IMAGE_DIR / "model_roc_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    with (RESULT_DIR / "logistic_metrics.json").open(encoding="utf-8") as file:
        logistic_metrics = json.load(file)
    comparison = pd.DataFrame(
        [logistic_metrics, forest_metrics],
        index=["Logistic regression", "Random forest"],
    )
    comparison.index.name = "model"
    comparison.to_csv(RESULT_DIR / "model_comparison.csv")

    chart_data = comparison[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    chart_data.plot(kind="bar", figsize=(10, 5.5), color=[BLUE, ORANGE, "#62A87C", "#8E6BBE", "#D6A84B"])
    plt.title("Held-Out Test Performance", weight="bold", pad=14)
    plt.ylabel("Score")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    plt.legend(loc="lower right", ncol=3)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "model_metric_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    required_paths = [
        DATA_PATH,
        MODEL_DIR / "logistic_regression.joblib",
        RESULT_DIR / "logistic_metrics.json",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required project artifacts are missing: {missing}")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    data = pd.read_csv(DATA_PATH)
    x_train, x_test, y_train, y_test = split_data(data)
    search = tune_model(x_train, y_train)
    model = search.best_estimator_

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = evaluate(y_test, predictions, probabilities)

    joblib.dump(model, MODEL_DIR / "random_forest.joblib")
    with (RESULT_DIR / "random_forest_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)
    with (RESULT_DIR / "random_forest_best_params.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(search.best_params_, file, indent=2)

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results.to_csv(RESULT_DIR / "random_forest_cv_results.csv", index=False)
    importance = save_feature_importance(model, list(x_train.columns))
    save_confusion_matrix(y_test, predictions)
    save_model_comparison(y_test, probabilities, metrics, x_test)

    print(f"Best cross-validation ROC-AUC: {search.best_score_:.3f}")
    print(f"Best parameters: {search.best_params_}")
    print("\nHeld-out test metrics")
    for name, value in metrics.items():
        print(f"{name:>18}: {value:.3f}")
    print("\nFeature importance")
    print(importance.to_string(index=False, formatters={"importance": "{:.3f}".format}))


if __name__ == "__main__":
    main()
