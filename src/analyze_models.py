"""Explain model predictions and evaluate business-aware decision thresholds."""

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
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split


DATA_PATH = PROJECT_ROOT / "data" / "credit_risk_data.csv"
IMAGE_DIR = PROJECT_ROOT / "images"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"
TARGET = "default"
RANDOM_STATE = 42
FALSE_NEGATIVE_COST = 5
FALSE_POSITIVE_COST = 1

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


def calculate_permutation_importance(
    name: str,
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "model": name,
            "feature": x_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return importance


def plot_permutation_importance(all_importance: pd.DataFrame) -> None:
    ordered_features = (
        all_importance.groupby("feature")["importance_mean"]
        .mean()
        .sort_values()
        .index
    )
    plot_data = all_importance.copy()
    plot_data["feature"] = pd.Categorical(
        plot_data["feature"], categories=ordered_features, ordered=True
    )

    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=plot_data,
        x="importance_mean",
        y="feature",
        hue="model",
        palette=[BLUE, ORANGE],
    )
    plt.axvline(0, color="gray", linewidth=0.8)
    plt.title("Held-Out Permutation Importance", weight="bold", pad=14)
    plt.xlabel("Decrease in ROC-AUC after shuffling")
    plt.ylabel("")
    plt.legend(title="")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        IMAGE_DIR / "permutation_importance_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def out_of_fold_probabilities(model, x_train, y_train) -> np.ndarray:
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(
        clone(model),
        x_train,
        y_train,
        cv=folds,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]


def threshold_table(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.10, 0.81, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "accuracy": accuracy_score(y_true, predictions),
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1": f1_score(y_true, predictions, zero_division=0),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "cost": int(
                    FALSE_POSITIVE_COST * fp + FALSE_NEGATIVE_COST * fn
                ),
            }
        )
    return pd.DataFrame(rows)


def test_metrics_at_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "cost": int(FALSE_POSITIVE_COST * fp + FALSE_NEGATIVE_COST * fn),
    }


def plot_threshold_tradeoff(
    model_name: str,
    table: pd.DataFrame,
    chosen_threshold: float,
    color: str,
) -> None:
    plt.figure(figsize=(8, 5.5))
    plt.plot(table["threshold"], table["precision"], label="Precision", color=BLUE)
    plt.plot(table["threshold"], table["recall"], label="Recall", color=ORANGE)
    plt.plot(table["threshold"], table["f1"], label="F1", color="#62A87C")
    plt.axvline(
        chosen_threshold,
        color=color,
        linestyle="--",
        label=f"Chosen threshold ({chosen_threshold:.2f})",
    )
    plt.title(f"{model_name} Threshold Trade-off", weight="bold", pad=14)
    plt.xlabel("Classification threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    filename = model_name.lower().replace(" ", "_") + "_threshold_tradeoff.png"
    plt.savefig(IMAGE_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    x_train, x_test, y_train, y_test = split_data(data)
    models = {
        "Logistic regression": joblib.load(
            MODEL_DIR / "logistic_regression.joblib"
        ),
        "Random forest": joblib.load(MODEL_DIR / "random_forest.joblib"),
    }

    importance_frames = []
    threshold_results = {}

    for index, (name, model) in enumerate(models.items()):
        importance_frames.append(
            calculate_permutation_importance(name, model, x_test, y_test)
        )

        training_probabilities = out_of_fold_probabilities(model, x_train, y_train)
        training_thresholds = threshold_table(y_train, training_probabilities)
        chosen_row = training_thresholds.loc[training_thresholds["cost"].idxmin()]
        chosen_threshold = float(chosen_row["threshold"])

        test_probabilities = model.predict_proba(x_test)[:, 1]
        threshold_results[name] = test_metrics_at_threshold(
            y_test, test_probabilities, chosen_threshold
        )
        training_thresholds.to_csv(
            RESULT_DIR / f"{name.lower().replace(' ', '_')}_threshold_search.csv",
            index=False,
        )
        plot_threshold_tradeoff(
            name,
            training_thresholds,
            chosen_threshold,
            [BLUE, ORANGE][index],
        )

    all_importance = pd.concat(importance_frames, ignore_index=True)
    all_importance.to_csv(RESULT_DIR / "permutation_importance.csv", index=False)
    plot_permutation_importance(all_importance)

    with (RESULT_DIR / "threshold_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "cost_assumption": {
                    "false_negative": FALSE_NEGATIVE_COST,
                    "false_positive": FALSE_POSITIVE_COST,
                },
                "test_results": threshold_results,
            },
            file,
            indent=2,
        )

    print("Permutation importance (top five per model)")
    for name in models:
        print(f"\n{name}")
        subset = all_importance[all_importance["model"] == name].head(5)
        print(
            subset[["feature", "importance_mean"]].to_string(
                index=False,
                formatters={"importance_mean": "{:.3f}".format},
            )
        )

    print("\nBusiness-threshold test results")
    for name, metrics in threshold_results.items():
        print(
            f"{name}: threshold={metrics['threshold']:.2f}, "
            f"precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, "
            f"cost={metrics['cost']}"
        )


if __name__ == "__main__":
    main()
