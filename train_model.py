"""
train_model.py
---------------
Loads the admission dataset, builds a preprocessing + ML pipeline, trains and
compares several classifiers, evaluates them, and saves the best pipeline to
trained_model.pkl for later use by predict_gui.py / predict_cli.py.
"""

import joblib
import matplotlib
matplotlib.use("Agg")  # so it works headlessly, saving PNGs instead of popping a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "admission_dataset.csv"
TARGET = "admission_status"

NUMERIC_FEATURES = [
    "age", "utme_score", "post_utme_score", "olevel_points",
    "subject_relevance", "interview_score",
]
CATEGORICAL_FEATURES = ["catchment_category", "course_choice"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


def build_preprocessor():
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def get_candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    models = get_candidate_models()

    results = {}
    fitted_pipelines = {}

    print("=" * 60)
    print("MODEL TRAINING & COMPARISON")
    print("=" * 60)

    for name, clf in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results[name] = {"accuracy": acc, "f1": f1, "roc_auc": auc}
        fitted_pipelines[name] = pipeline

        print(f"\n--- {name} ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Rejected", "Admitted"]))

    # ---- Pick the best model by F1-score --------------------------------
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_pipeline = fitted_pipelines[best_name]
    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_name}  (F1={results[best_name]['f1']:.4f})")
    print("=" * 60)

    # ---- Save comparison chart -------------------------------------------
    comp_df = pd.DataFrame(results).T
    comp_df.plot(kind="bar", figsize=(8, 5), rot=20)
    plt.title("Model Comparison (Accuracy / F1 / ROC-AUC)")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.close()

    # ---- Save confusion matrix of the best model --------------------------
    y_pred_best = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rejected", "Admitted"],
                yticklabels=["Rejected", "Admitted"])
    plt.title(f"Confusion Matrix - {best_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    # ---- Persist the winning pipeline -------------------------------------
    joblib.dump({"pipeline": best_pipeline, "model_name": best_name}, "trained_model.pkl")
    print("\nSaved best pipeline -> trained_model.pkl")
    print("Saved charts -> model_comparison.png, confusion_matrix.png")


if __name__ == "__main__":
    main()
