import os
import json
import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "bc_dataset", "data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
INTERNAL_TEST_PATH = os.path.join(BASE_DIR, "internal_test_data.pkl")

# add or remove entries here to change which models get trained
MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=1),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=1),
    "svm": SVC(probability=True, random_state=1),
}

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "svm": "SVM",
}


def clean_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(['id', 'Unnamed: 32'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


def train_all_models(data):
    """
    Trains every model in MODEL_REGISTRY on the same train/test split,
    saves each model as its own pickle, saves one shared scaler, and
    writes a metrics.json summarising performance per model.
    """
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    X_temp, X_validation, y_temp, y_validation = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=1,
        stratify=y_temp
    )

    # Save the internal test set
    with open(INTERNAL_TEST_PATH, 'wb') as f:
        pickle.dump(
            {
                "X_test": X_test,
                "y_test": y_test
            },
            f
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the simulated external validation set
    VALIDATION_PATH = os.path.join(
        BASE_DIR,
        "validation_data.pkl"
    )

    with open(VALIDATION_PATH, 'wb') as f:
        pickle.dump(
            {
                "X_validation": X_validation,
                "y_validation": y_validation
            },
            f
        )

    os.makedirs(MODELS_DIR, exist_ok=True)
    all_metrics = {}

    for key, model in MODEL_REGISTRY.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]  # prob of malignant, for ROC-AUC

        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics = {
            "display_name": DISPLAY_NAMES[key],
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        }
        all_metrics[key] = metrics

        model_path = os.path.join(MODELS_DIR, f"{key}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"{DISPLAY_NAMES[key]}: accuracy={metrics['accuracy']}, f1={metrics['f1_score']}, roc_auc={metrics['roc_auc']}")

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    with open(METRICS_PATH, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics

def save_feature_importance(data):
    X = data.drop('diagnosis', axis=1)

    feature_importance = {}
    model_path = os.path.join(
        MODELS_DIR,
        "logistic_regression.pkl"
    )

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    feature_importance["Logistic Regression"] = {
        feature: float(abs(coef))
        for feature, coef in zip(
            X.columns,
            model.coef_[0]
        )
    }

    # -------------------------------
    # Random Forest
    # -------------------------------
    model_path = os.path.join(
        MODELS_DIR,
        "random_forest.pkl"
    )

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    feature_importance["Random Forest"] = {
        feature: float(importance)
        for feature, importance in zip(
            X.columns,
            model.feature_importances_
        )
    }

    
    feature_importance["SVM"] = {}

    FEATURE_IMPORTANCE_PATH = os.path.join(
        BASE_DIR,
        "feature_importance.json"
    )

    with open(
        FEATURE_IMPORTANCE_PATH,
        'w'
    ) as f:
        json.dump(
            feature_importance,
            f,
            indent=2
        )

    print(
        f"Saved feature importance to "
        f"{FEATURE_IMPORTANCE_PATH}"
    )


def main():
    data = clean_data()
    train_all_models(data)
    save_feature_importance(data)

    print(f"\nSaved models to {MODELS_DIR}/")
    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()


#python -m streamlit run app\main.py