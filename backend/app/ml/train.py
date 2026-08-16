import os
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, average_precision_score
)

from backend.app.ml.preprocess import prepare_data


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "diabetic_data.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _class_weight_ratio(y_train) -> float:
    """Return scale_pos_weight for XGBoost (neg/pos ratio)."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    return float(neg) / float(pos) if pos > 0 else 1.0


def train_and_evaluate(csv_path: str = str(DATA_PATH)):
    """
    Train Logistic Regression, Random Forest, and XGBoost with:
      - Full dataset
      - Engineered features
      - Class-imbalance handling
      - Tuned hyperparameters

    Best model is selected by ROC-AUC and persisted.
    """

    print("Preparing dataset (full data, deduped, feature-engineered)...")

    X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols = (
        prepare_data(csv_path)
    )

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Positive rate (train): {y_train.mean():.3f}")

    # Fit preprocessing pipeline on training data only
    print("Preprocessing data...")

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Get feature names after one-hot encoding
    cat_encoder = (
        preprocessor
        .named_transformers_["cat"]
        .named_steps["onehot"]
    )

    encoded_cat_cols = list(
        cat_encoder.get_feature_names_out(cat_cols)
    )

    feature_names = list(num_cols) + encoded_cat_cols

    spw = _class_weight_ratio(y_train)

    print(
        f"Class imbalance scale_pos_weight "
        f"(neg/pos): {spw:.2f}"
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            C=0.1,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="aucpr",
            random_state=42
        ),
    }

    results = {}

    best_auc = -1
    best_model_name = None
    best_model_obj = None

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train_proc, y_train)

        y_pred_prob = model.predict_proba(X_test_proc)[:, 1]
        y_pred = model.predict(X_test_proc)

        auc = roc_auc_score(y_test, y_pred_prob)
        pr_auc = average_precision_score(y_test, y_pred_prob)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(
            y_test, y_pred, zero_division=0
        )
        rec = recall_score(
            y_test, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        results[name] = {
            "roc_auc": round(float(auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
        }

        print(
            f"  ROC-AUC: {auc:.4f} | "
            f"PR-AUC: {pr_auc:.4f} | "
            f"F1: {f1:.4f} | "
            f"Recall: {rec:.4f}"
        )

        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model_obj = model

    print(
        f"\nBest model: {best_model_name} "
        f"(ROC-AUC: {best_auc:.4f})"
    )

    # ========================================================
    # SAVE MODELS
    # ========================================================

    print(f"Saving models to: {MODELS_DIR}")

    joblib.dump(
        preprocessor,
        MODELS_DIR / "preprocessor.pkl"
    )

    joblib.dump(
        best_model_obj,
        MODELS_DIR / "best_model.pkl"
    )

    metadata = {
        "best_model": best_model_name,
        "metrics": results,
        "feature_names": feature_names,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }

    with open(
        MODELS_DIR / "model_metadata.json",
        "w"
    ) as f:
        json.dump(metadata, f, indent=2)

    print("ML Pipeline training complete.")

    return metadata


if __name__ == "__main__":
    train_and_evaluate()