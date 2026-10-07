import os
import pickle
import pandas as pd
import streamlit as st
import plotly.express as px
import shap

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


st.set_page_config(
    page_title="Validation XAI Analysis",
    page_icon="🧠",
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
INTERNAL_TEST_PATH = os.path.join(
    MODEL_DIR,
    "internal_test_data.pkl"
)
VALIDATION_PATH = os.path.join(
    MODEL_DIR,
    "validation_data.pkl"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_test_data():

    with open(INTERNAL_TEST_PATH, "rb") as f:
        data = pickle.load(f)

    return data["X_test"], data["y_test"]


@st.cache_data
def load_validation_data():

    with open(VALIDATION_PATH, "rb") as f:
        data = pickle.load(f)

    return data["X_validation"], data["y_validation"]


@st.cache_resource
def load_scaler():

    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_model(model_name):

    path = os.path.join(
        MODELS_DIR,
        f"{model_name}.pkl"
    )

    with open(path, "rb") as f:
        return pickle.load(f)


X_test, y_test = load_test_data()
X_validation, y_validation = load_validation_data()

scaler = load_scaler()


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("🧠 Validation XAI Analysis")

st.write(
    "This page compares model performance on the internal test set "
    "and the simulated external validation set, then uses SHAP to "
    "help explain the model's behavior on unseen validation data."
)

st.divider()


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

models = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "SVM": "svm"
}

selected_model_name = st.selectbox(
    "Select Model",
    list(models.keys())
)

model = load_model(
    models[selected_model_name]
)


# --------------------------------------------------
# SCALE DATA
# --------------------------------------------------

X_test_scaled = scaler.transform(X_test)
X_validation_scaled = scaler.transform(X_validation)


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

y_validation_pred = model.predict(X_validation_scaled)
y_validation_proba = model.predict_proba(
    X_validation_scaled
)[:, 1]


# --------------------------------------------------
# METRICS FUNCTION
# --------------------------------------------------

def calculate_metrics(y_true, y_pred, y_proba):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba)
    }


test_metrics = calculate_metrics(
    y_test,
    y_test_pred,
    y_test_proba
)

validation_metrics = calculate_metrics(
    y_validation,
    y_validation_pred,
    y_validation_proba
)


# --------------------------------------------------
# PERFORMANCE COMPARISON
# --------------------------------------------------

st.subheader("Internal Test vs Simulated External Validation")

comparison_data = []

for metric in test_metrics:

    comparison_data.append({
        "Metric": metric,
        "Internal Test": test_metrics[metric],
        "External Validation": validation_metrics[metric],
        "Change": (
            validation_metrics[metric]
            - test_metrics[metric]
        )
    })

comparison_df = pd.DataFrame(comparison_data)

display_df = comparison_df.copy()

display_df["Internal Test"] = display_df[
    "Internal Test"
].round(4)

display_df["External Validation"] = display_df[
    "External Validation"
].round(4)

display_df["Change"] = display_df[
    "Change"
].round(4)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# PERFORMANCE CHART
# --------------------------------------------------

fig_data = comparison_df.melt(
    id_vars="Metric",
    value_vars=[
        "Internal Test",
        "External Validation"
    ],
    var_name="Dataset",
    value_name="Score"
)

fig = px.bar(
    fig_data,
    x="Metric",
    y="Score",
    color="Dataset",
    barmode="group",
    text="Score",
    title=f"{selected_model_name} Performance Comparison"
)

fig.update_traces(
    texttemplate="%{text:.3f}",
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
# SIMPLE PERFORMANCE EXPLANATION
# --------------------------------------------------

st.subheader("📖 What happened to the performance?")

accuracy_change = (
    validation_metrics["Accuracy"]
    - test_metrics["Accuracy"]
)

if accuracy_change < -0.01:

    st.warning(
        f"The model's accuracy decreased by "
        f"{abs(accuracy_change):.4f} on the simulated "
        f"external validation set."
    )

    st.write(
        "This means the model performed better on the internal "
        "test samples than on the unseen validation samples. "
        "This can happen because the validation samples contain "
        "feature patterns that are somewhat different from the "
        "patterns the model learned during training."
    )

elif accuracy_change > 0.01:

    st.success(
        f"The model's accuracy increased by "
        f"{accuracy_change:.4f} on the simulated external "
        f"validation set."
    )

    st.write(
        "The model performed slightly better on the held-out "
        "validation samples than on the internal test samples. "
        "Differences like this can occur because each partition "
        "contains a different combination of cases."
    )

else:

    st.info(
        "The model's accuracy remained relatively stable between "
        "the internal test set and the simulated external "
        "validation set."
    )

    st.write(
        "This suggests that the model's overall classification "
        "performance was relatively consistent across the two "
        "unseen partitions."
    )


st.divider()


# --------------------------------------------------
# ERROR ANALYSIS
# --------------------------------------------------

st.subheader("🔎 Validation Error Analysis")

validation_errors = y_validation_pred != y_validation.values

error_count = int(validation_errors.sum())
correct_count = int((~validation_errors).sum())

col1, col2 = st.columns(2)

col1.metric(
    "Correct Predictions",
    correct_count
)

col2.metric(
    "Incorrect Predictions",
    error_count
)

if error_count > 0:

    st.write(
        "The model made some incorrect predictions on the "
        "validation set. SHAP can help us inspect which features "
        "influenced those predictions."
    )

else:

    st.success(
        "The model correctly classified every validation sample."
    )


st.divider()


# --------------------------------------------------
# SHAP EXPLANATION
# --------------------------------------------------

st.subheader("🧠 SHAP Explanation of Validation Predictions")

st.write(
    "SHAP explains how individual features influenced the model's "
    "prediction. A positive SHAP value pushes the prediction toward "
    "malignant, while a negative value pushes it toward benign."
)


# --------------------------------------------------
# CREATE SHAP EXPLAINER
# --------------------------------------------------

if selected_model_name == "Logistic Regression":

    explainer = shap.LinearExplainer(
        model,
        X_test_scaled
    )

    shap_values = explainer(
        X_validation_scaled
    )

elif selected_model_name == "Random Forest":

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(
        X_validation_scaled
    )

else:

    st.info(
        "For the SVM model, this page uses a model-agnostic "
        "SHAP explainer. This may take a little longer."
    )

    background = shap.sample(
        X_test_scaled,
        min(50, len(X_test_scaled)),
        random_state=1
    )

    explainer = shap.KernelExplainer(
        model.predict_proba,
        background
    )

    shap_values = explainer.shap_values(
        X_validation_scaled[:20]
    )


# --------------------------------------------------
# HANDLE SHAP OUTPUT
# --------------------------------------------------

if selected_model_name == "SVM":

    if isinstance(shap_values, list):

        shap_array = shap_values[1]

    else:

        shap_array = shap_values

else:

    shap_array = shap_values.values

    if len(shap_array.shape) == 3:

        shap_array = shap_array[:, :, 1]


# --------------------------------------------------
# GLOBAL VALIDATION SHAP
# --------------------------------------------------

st.subheader("Global Feature Influence on Validation Predictions")

mean_abs_shap = abs(shap_array).mean(axis=0)

shap_importance = pd.DataFrame({
    "Feature": X_validation.columns,
    "Mean Absolute SHAP": mean_abs_shap
})

shap_importance = shap_importance.sort_values(
    "Mean Absolute SHAP",
    ascending=False
).head(10)

fig = px.bar(
    shap_importance.sort_values(
        "Mean Absolute SHAP"
    ),
    x="Mean Absolute SHAP",
    y="Feature",
    orientation="h",
    title="Top Features Influencing Validation Predictions"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# SIMPLE XAI EXPLANATION
# --------------------------------------------------

st.subheader("💡 SHAP in Simple English")

top_features = shap_importance.head(5)["Feature"].tolist()

feature_text = ", ".join(top_features)

st.write(
    f"For this model, the features with the strongest influence "
    f"on validation predictions include **{feature_text}**. "
    "This means these measurements had the greatest influence "
    "on how the model separated benign and malignant predictions "
    "within the validation samples."
)

st.caption(
    "SHAP describes the model's behavior and feature influence. "
    "It does not prove that a feature biologically causes cancer."
)


st.divider()


# --------------------------------------------------
# INDIVIDUAL VALIDATION CASE
# --------------------------------------------------

st.subheader("🔬 Explain an Individual Validation Prediction")

sample_index = st.number_input(
    "Validation Sample Index",
    min_value=0,
    max_value=len(X_validation) - 1,
    value=0,
    step=1
)

sample_prediction = y_validation_pred[sample_index]
sample_probability = y_validation_proba[sample_index]
sample_actual = y_validation.iloc[sample_index]


col1, col2, col3 = st.columns(3)

col1.metric(
    "Actual",
    "Malignant" if sample_actual == 1 else "Benign"
)

col2.metric(
    "Predicted",
    "Malignant" if sample_prediction == 1 else "Benign"
)

col3.metric(
    "Malignant Probability",
    f"{sample_probability:.2%}"
)


# --------------------------------------------------
# SAMPLE SHAP VALUES
# --------------------------------------------------

sample_shap = shap_array[sample_index]

sample_explanation = pd.DataFrame({
    "Feature": X_validation.columns,
    "SHAP Value": sample_shap,
    "Absolute Influence": abs(sample_shap)
})

sample_explanation = sample_explanation.sort_values(
    "Absolute Influence",
    ascending=False
)

st.dataframe(
    sample_explanation.head(10),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# SIMPLE INDIVIDUAL EXPLANATION
# --------------------------------------------------

top_positive = sample_explanation[
    sample_explanation["SHAP Value"] > 0
].head(3)

top_negative = sample_explanation[
    sample_explanation["SHAP Value"] < 0
].head(3)


if sample_prediction == 1:

    prediction_direction = "malignant"

else:

    prediction_direction = "benign"


st.write(
    f"The model predicted this validation sample as "
    f"**{prediction_direction}**."
)

if not top_positive.empty:

    positive_features = ", ".join(
        top_positive["Feature"].tolist()
    )

    st.write(
        f"Features pushing the prediction toward malignant "
        f"include **{positive_features}**."
    )

if not top_negative.empty:

    negative_features = ", ".join(
        top_negative["Feature"].tolist()
    )

    st.write(
        f"Features pushing the prediction toward benign "
        f"include **{negative_features}**."
    )


if sample_prediction == sample_actual:

    st.success(
        "The model's prediction matches the actual diagnosis. "
        "The SHAP values above show which features contributed "
        "most strongly to that prediction."
    )

else:

    st.warning(
        "This prediction is incorrect. The SHAP values help show "
        "which feature pattern influenced the model toward the "
        "wrong class."
    )


st.caption(
    "Important: SHAP explains why the machine-learning model "
    "made a prediction. It should not be interpreted as a "
    "medical diagnosis or causal explanation."
)