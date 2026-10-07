import os
import pickle
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


st.set_page_config(
    page_title="Simulated External Validation",
    page_icon="🧪",
    layout="wide"
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_DIR = os.path.join(PROJECT_DIR, "Model")
MODELS_DIR = os.path.join(MODEL_DIR, "models")

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
VALIDATION_PATH = os.path.join(MODEL_DIR, "validation_data.pkl")


# --------------------------------------------------
# LOAD VALIDATION DATA
# --------------------------------------------------

@st.cache_data
def load_validation_data():
    with open(VALIDATION_PATH, "rb") as f:
        validation_data = pickle.load(f)

    X_validation = validation_data["X_validation"]
    y_validation = validation_data["y_validation"]

    return X_validation, y_validation


@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_model(model_name):
    model_path = os.path.join(
        MODELS_DIR,
        f"{model_name}.pkl"
    )

    with open(model_path, "rb") as f:
        return pickle.load(f)


X_validation, y_validation = load_validation_data()
scaler = load_scaler()


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("🧪 Simulated External Validation")

st.write(
    "Evaluation of the trained models on a completely held-out "
    "validation partition from the same Breast Cancer Wisconsin dataset."
)

st.divider()


# --------------------------------------------------
# VALIDATION INFORMATION
# --------------------------------------------------

st.subheader("Validation Set Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Validation Samples",
    len(X_validation)
)

col2.metric(
    "Benign Cases",
    int((y_validation == 0).sum())
)

col3.metric(
    "Malignant Cases",
    int((y_validation == 1).sum())
)

st.info(
    "This is simulated external validation. The validation samples "
    "were separated before model training and were not used during "
    "model fitting or internal testing."
)

st.divider()


# --------------------------------------------------
# SCALE VALIDATION DATA
# --------------------------------------------------

X_validation_scaled = scaler.transform(X_validation)


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

models = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "SVM": "svm"
}

results = []
confusion_matrices = {}

for display_name, model_name in models.items():

    model = load_model(model_name)

    y_pred = model.predict(X_validation_scaled)
    y_proba = model.predict_proba(X_validation_scaled)[:, 1]

    accuracy = accuracy_score(
        y_validation,
        y_pred
    )

    precision = precision_score(
        y_validation,
        y_pred
    )

    recall = recall_score(
        y_validation,
        y_pred
    )

    f1 = f1_score(
        y_validation,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_validation,
        y_proba
    )

    results.append({
        "Model": display_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4)
    })

    confusion_matrices[display_name] = confusion_matrix(
        y_validation,
        y_pred
    )


results_df = pd.DataFrame(results)


# --------------------------------------------------
# RESULTS TABLE
# --------------------------------------------------

st.subheader("External Validation Performance")

st.dataframe(
    results_df,
    use_container_width=True,
    hide_index=True
)

st.divider()


# --------------------------------------------------
# PERFORMANCE COMPARISON
# --------------------------------------------------

st.subheader("Model Performance Comparison")

metric = st.selectbox(
    "Select Metric",
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
)

fig = px.bar(
    results_df,
    x="Model",
    y=metric,
    title=f"{metric} on Simulated External Validation",
    text=metric
)

fig.update_traces(
    textposition="outside"
)

fig.update_yaxes(
    range=[0, 1]
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()


# --------------------------------------------------
# CONFUSION MATRICES
# --------------------------------------------------

st.subheader("Confusion Matrices")

selected_model = st.selectbox(
    "Select Model",
    list(models.keys())
)

cm = confusion_matrices[selected_model]

cm_df = pd.DataFrame(
    cm,
    index=["Actual Benign", "Actual Malignant"],
    columns=["Predicted Benign", "Predicted Malignant"]
)

fig = px.imshow(
    cm_df,
    text_auto=True,
    title=f"{selected_model} - Confusion Matrix"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()


# --------------------------------------------------
# INTERPRETATION
# --------------------------------------------------

st.subheader("Interpretation")

best_model = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

best_f1 = results_df["F1 Score"].max()

st.write(
    f"Based on F1 Score, **{best_model}** achieved the strongest "
    f"performance on the simulated external validation set "
    f"with an F1 Score of **{best_f1:.4f}**."
)

st.caption(
    "Note: Because this validation partition comes from the same "
    "Breast Cancer Wisconsin dataset rather than a separate "
    "independent dataset, this should be interpreted as simulated "
    "external validation rather than true external clinical validation."
)