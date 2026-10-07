import os
import pickle
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Individual SHAP Explanation",
    page_icon="🔍",
    layout="wide"
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "bc_dataset",
    "data.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "Model",
    "models"
)

SCALER_PATH = os.path.join(
    PROJECT_DIR,
    "Model",
    "scaler.pkl"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_PATH)

    data = data.drop(
        ["id", "Unnamed: 32"],
        axis=1
    )

    data["diagnosis"] = data["diagnosis"].map({
        "M": 1,
        "B": 0
    })

    return data


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model(model_name):

    model_path = os.path.join(
        MODEL_DIR,
        f"{model_name}.pkl"
    )

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model


# --------------------------------------------------
# LOAD SCALER
# --------------------------------------------------

@st.cache_resource
def load_scaler():

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    return scaler


# --------------------------------------------------
# LOAD DATA + SCALER
# --------------------------------------------------

data = load_data()
scaler = load_scaler()

X = data.drop(
    "diagnosis",
    axis=1
)

X_scaled = scaler.transform(X)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("🔍 Individual SHAP Explanation")

st.write(
    "This page explains why a trained model made a prediction "
    "for an individual sample from the dataset."
)

st.info(
    "SHAP explains the model's reasoning for this prediction. "
    "It does not provide a medical diagnosis or indicate causation."
)

st.divider()


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

st.subheader("Select Model")

selected_model = st.selectbox(
    "Choose the model",
    [
        "Logistic Regression",
        "Random Forest"
    ]
)

if selected_model == "Logistic Regression":
    model_file = "logistic_regression"
else:
    model_file = "random_forest"

model = load_model(model_file)


# --------------------------------------------------
# SAMPLE SELECTION
# --------------------------------------------------

st.subheader("Select Patient Sample")

sample_index = st.number_input(
    "Dataset sample number",
    min_value=0,
    max_value=len(X) - 1,
    value=0,
    step=1
)

sample = X.iloc[[sample_index]]
sample_scaled = X_scaled_df.iloc[[sample_index]]


# --------------------------------------------------
# ACTUAL DIAGNOSIS
# --------------------------------------------------

actual_value = data.iloc[sample_index]["diagnosis"]

if actual_value == 1:
    actual_label = "Malignant"
else:
    actual_label = "Benign"


# --------------------------------------------------
# MODEL PREDICTION
# --------------------------------------------------

prediction = model.predict(sample_scaled)[0]

probability = model.predict_proba(
    sample_scaled
)[0][1]


if prediction == 1:
    prediction_label = "Malignant"
else:
    prediction_label = "Benign"


# --------------------------------------------------
# DISPLAY PREDICTION
# --------------------------------------------------

st.divider()

st.subheader("Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model Prediction",
        prediction_label
    )

with col2:
    st.metric(
        "Malignant Probability",
        f"{probability * 100:.2f}%"
    )

with col3:
    st.metric(
        "Actual Dataset Label",
        actual_label
    )


# --------------------------------------------------
# SHAP EXPLANATION
# --------------------------------------------------

st.divider()

st.subheader("Why did the model make this prediction?")

with st.spinner("Calculating SHAP explanation..."):

    if selected_model == "Logistic Regression":

        explainer = shap.LinearExplainer(
            model,
            X_scaled_df
        )

        shap_values = explainer(
            sample_scaled
        )

    else:

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer(
            sample_scaled
        )


# --------------------------------------------------
# SHAP WATERFALL PLOT
# --------------------------------------------------

st.write(
    "The waterfall plot shows how individual features "
    "contributed to this specific prediction."
)

fig = plt.figure()

shap.plots.waterfall(
    shap_values[0],
    max_display=10,
    show=False
)

st.pyplot(
    fig,
    clear_figure=True
)

plt.close(fig)


# --------------------------------------------------
# FEATURE CONTRIBUTIONS
# --------------------------------------------------

st.divider()

st.subheader("Feature Contributions")

feature_contributions = pd.DataFrame({
    "Feature": X.columns,
    "SHAP Value": shap_values.values[0]
})

feature_contributions["Absolute SHAP"] = (
    feature_contributions["SHAP Value"].abs()
)

feature_contributions = feature_contributions.sort_values(
    "Absolute SHAP",
    ascending=False
)

feature_contributions = feature_contributions.drop(
    "Absolute SHAP",
    axis=1
)

feature_contributions["SHAP Value"] = (
    feature_contributions["SHAP Value"].round(6)
)

st.dataframe(
    feature_contributions,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# TOP CONTRIBUTING FEATURES
# --------------------------------------------------

st.divider()

st.subheader("Top Contributing Features")

top_features = feature_contributions.head(5)

for _, row in top_features.iterrows():

    shap_value = row["SHAP Value"]

    if shap_value > 0:

        direction = "toward malignant"

    else:

        direction = "toward benign"

    st.write(
        f"**{row['Feature']}**: "
        f"{shap_value:.4f} → {direction}"
    )


# --------------------------------------------------
# INTERPRETATION
# --------------------------------------------------

st.divider()

st.subheader("How to Interpret This Explanation")

st.write(
    "Positive SHAP values push the model's prediction "
    "toward malignant."
)

st.write(
    "Negative SHAP values push the model's prediction "
    "toward benign."
)

st.write(
    "A larger absolute SHAP value means that the feature "
    "had a stronger influence on this particular prediction."
)

st.warning(
    "These explanations describe model behavior only. "
    "They should not be interpreted as medical causation "
    "or as a substitute for clinical diagnosis."
)