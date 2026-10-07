import os
import pickle
import pandas as pd
import streamlit as st
import shap
import plotly.express as px
import matplotlib.pyplot as plt


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SHAP Explainability",
    page_icon="🧠",
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
# LOAD EVERYTHING
# --------------------------------------------------

data = load_data()
scaler = load_scaler()


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("🧠 SHAP Explainability")

st.write(
    "SHAP (SHapley Additive exPlanations) explains how "
    "individual features influence the predictions made by "
    "the machine-learning models."
)

st.divider()


# --------------------------------------------------
# EXPLANATION
# --------------------------------------------------

st.subheader("What is SHAP?")

st.write(
    "SHAP explains model predictions by assigning each feature "
    "a contribution value. A positive SHAP value pushes the "
    "prediction toward malignant, while a negative SHAP value "
    "pushes the prediction toward benign."
)

st.info(
    "SHAP explains the behavior of the machine-learning model. "
    "It does not mean that a feature causes cancer."
)

st.divider()


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

st.subheader("Select Model")

selected_model = st.selectbox(
    "Choose a model to explain",
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
# PREPARE FEATURES
# --------------------------------------------------

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
# CREATE SHAP EXPLAINER
# --------------------------------------------------

with st.spinner("Calculating SHAP values..."):

    if selected_model == "Logistic Regression":

        explainer = shap.LinearExplainer(
            model,
            X_scaled_df
        )

        shap_values = explainer(
            X_scaled_df
        )

    else:

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer(
            X_scaled_df
        )


# --------------------------------------------------
# GLOBAL SHAP IMPORTANCE
# --------------------------------------------------

st.subheader("Global SHAP Feature Importance")

st.write(
    "This ranking shows which features have the greatest "
    "overall influence on the model's predictions."
)

mean_abs_shap = (
    abs(shap_values.values)
    .mean(axis=0)
)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Mean |SHAP Value|": mean_abs_shap
})

importance_df = importance_df.sort_values(
    "Mean |SHAP Value|",
    ascending=False
)

importance_df = importance_df.reset_index(
    drop=True
)

importance_df.insert(
    0,
    "Rank",
    range(1, len(importance_df) + 1)
)


# --------------------------------------------------
# TOP N SLIDER
# --------------------------------------------------

top_n = st.slider(
    "Number of features to display",
    min_value=5,
    max_value=30,
    value=10,
    step=5
)

top_features = importance_df.head(top_n)


# --------------------------------------------------
# BAR CHART
# --------------------------------------------------

fig = px.bar(
    top_features.sort_values(
        "Mean |SHAP Value|",
        ascending=True
    ),
    x="Mean |SHAP Value|",
    y="Feature",
    orientation="h",
    title=f"Top {top_n} SHAP Features - {selected_model}",
    text="Mean |SHAP Value|"
)

fig.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

fig.update_layout(
    height=max(450, top_n * 40)
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# SHAP SUMMARY PLOT
# --------------------------------------------------

st.divider()

st.subheader("SHAP Summary Plot")

st.write(
    "The summary plot shows both the importance and "
    "direction of feature influence across the dataset."
)

fig_shap, ax = plt.subplots()

shap.summary_plot(
    shap_values,
    X_scaled_df,
    show=False
)

st.pyplot(
    fig_shap,
    clear_figure=True
)

plt.close(fig_shap)


# --------------------------------------------------
# FEATURE RANKING TABLE
# --------------------------------------------------

st.divider()

st.subheader("SHAP Feature Ranking")

display_df = importance_df.copy()

display_df["Mean |SHAP Value|"] = (
    display_df["Mean |SHAP Value|"].round(6)
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# TOP 5 FEATURES
# --------------------------------------------------

st.divider()

st.subheader("Top 5 SHAP Features")

top_5 = importance_df.head(5)

cols = st.columns(5)

for index, row in top_5.iterrows():

    with cols[index]:

        st.metric(
            label=row["Feature"],
            value=f"{row['Mean |SHAP Value|']:.4f}"
        )


# --------------------------------------------------
# INTERPRETATION
# --------------------------------------------------

st.divider()

st.subheader("How to Interpret SHAP Values")

st.write(
    "A positive SHAP value indicates that the feature "
    "pushes the model's prediction toward malignant."
)

st.write(
    "A negative SHAP value indicates that the feature "
    "pushes the model's prediction toward benign."
)

st.write(
    "The larger the absolute SHAP value, the stronger "
    "the feature's influence on that prediction."
)

st.warning(
    "SHAP values describe model behavior and should not "
    "be interpreted as medical causation."
)