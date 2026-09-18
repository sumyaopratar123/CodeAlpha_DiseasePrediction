import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
SCREEN_DIR = os.path.join(ROOT, "screenshots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCREEN_DIR, exist_ok=True)

def get_models(random_state=42):
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=random_state),
        "SVM": SVC(probability=True, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=random_state,
            n_jobs=-1
        ),
    }

def make_pipeline(X, model):
    numeric = list(X.columns)
    prep = ColumnTransformer(
        [("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric)],
        remainder="drop"
    )
    return Pipeline([("preprocessor", prep), ("model", model)])

def evaluate_models(X_train, X_test, y_train, y_test, prefix):
    rows = []
    fitted = {}
    for name, model in get_models().items():
        pipe = make_pipeline(X_train, model)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]

        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, prob),
        }
        rows.append(row)
        fitted[name] = pipe

        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cbar=False)
        plt.title(f"{prefix} - {name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        safe = name.lower().replace(" ", "_")
        plt.savefig(os.path.join(SCREEN_DIR, f"{prefix.lower()}_{safe}_confusion_matrix.png"), dpi=160)
        plt.close()

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(SCREEN_DIR, f"{prefix.lower()}_model_comparison.csv"), index=False)

    plt.figure(figsize=(8, 6))
    for name, pipe in fitted.items():
        prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{prefix} - ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SCREEN_DIR, f"{prefix.lower()}_roc_curves.png"), dpi=160)
    plt.close()

    for name, pipe in fitted.items():
        safe = name.lower().replace(" ", "_")
        joblib.dump(pipe, os.path.join(MODEL_DIR, f"{prefix.lower()}_{safe}.joblib"))

    return results

def save_metadata(prefix, feature_names):
    with open(os.path.join(MODEL_DIR, f"{prefix.lower()}_features.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
