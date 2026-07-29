"""Train and evaluate a logistic-regression credit-risk baseline."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = PROJECT_ROOT / "data" / "credit_risk_data.csv"
IMAGE_DIR = PROJECT_ROOT / "images"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"
TARGET = "default"
RANDOM_STATE = 42

BLUE = "#3B82B8"
ORANGE = "#E07A3F"


def split_data(data: pd.DataFrame):
    """Create a reproducible stratified 80/20 train-test split."""
    features = data.drop(columns=TARGET)
    target = data[TARGET]
    return train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def build_pipeline() -> Pipeline:
    """Scale numeric features, then fit the classifier."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE),
            ),
        ]
    )


def evaluate(y_true: pd.Series, predictions, probabilities) -> dict[str, float]:
    """Calculate test-set classification metrics."""
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
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title("Logistic Regression Confusion Matrix", weight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(
        IMAGE_DIR / "logistic_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def save_roc_curve(y_true: pd.Series, probabilities, roc_auc: float) -> None:
    display = RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
        name=f"Logistic regression (AUC = {roc_auc:.3f})",
        curve_kwargs={"color": ORANGE},
    )
    display.ax_.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    display.ax_.set_title("Logistic Regression ROC Curve", weight="bold", pad=14)
    display.ax_.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "logistic_roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_coefficients(model: Pipeline, feature_names: list[str]) -> None:
    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.named_steps["classifier"].coef_[0],
        }
    )
    coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()
    coefficients = coefficients.sort_values("absolute_coefficient", ascending=False)
    coefficients.to_csv(RESULT_DIR / "logistic_coefficients.csv", index=False)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} does not exist. Run src/generate_data.py first."
        )

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    x_train, x_test, y_train, y_test = split_data(data)

    model = build_pipeline()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = evaluate(y_test, predictions, probabilities)

    joblib.dump(model, MODEL_DIR / "logistic_regression.joblib")
    with (RESULT_DIR / "logistic_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    save_coefficients(model, list(x_train.columns))
    save_confusion_matrix(y_test, predictions)
    save_roc_curve(y_test, probabilities, metrics["roc_auc"])

    print(f"Training applicants: {len(x_train):,}")
    print(f"Test applicants: {len(x_test):,}")
    print(f"Training default rate: {y_train.mean():.2%}")
    print(f"Test default rate: {y_test.mean():.2%}")
    print("\nHeld-out test metrics")
    for name, value in metrics.items():
        print(f"{name:>18}: {value:.3f}")


if __name__ == "__main__":
    main()
