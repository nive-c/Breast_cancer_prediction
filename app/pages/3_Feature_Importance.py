import os
import json

import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Feature Importance",
    page_icon="📌",
    layout="wide"
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROJECT_DIR = os.path.dirname(
    BASE_DIR
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "Model"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    MODEL_DIR,
    "feature_importance.json"
)


# ============================================================
# LOAD FEATURE IMPORTANCE
# ============================================================

@st.cache_data
def load_feature_importance():

    if not os.path.exists(
        FEATURE_IMPORTANCE_PATH
    ):

        return None

    with open(
        FEATURE_IMPORTANCE_PATH,
        "r"
    ) as file:

        return json.load(file)


feature_importance = (
    load_feature_importance()
)


# ============================================================
# CHECK FILE
# ============================================================

if feature_importance is None:

    st.error(
        "Feature importance file was not found."
    )

    st.info(
        "Please run Model/main.py first."
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title(
    "📌 Feature Importance"
)

st.write(
    "This page shows which features had the "
    "greatest influence on the trained models."
)

st.divider()


# ============================================================
# EXPLANATION
# ============================================================

st.subheader(
    "What is Feature Importance?"
)

st.write(
    "Feature importance indicates how strongly "
    "a model relies on individual input features "
    "when making predictions."
)

st.info(
    "Higher importance means the model relied more "
    "heavily on that feature. It does not mean that "
    "the feature causes cancer."
)

st.divider()


# ============================================================
# MODEL SELECTION
# ============================================================

available_models = [
    model
    for model in feature_importance
    if feature_importance[model]
]


if not available_models:

    st.warning(
        "No model-based feature importance is available."
    )

    st.stop()


selected_model = st.selectbox(
    "Select Model",
    available_models
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

importance_data = feature_importance[
    selected_model
]

importance_df = pd.DataFrame(
    importance_data.items(),
    columns=[
        "Feature",
        "Importance"
    ]
)


# Sort by importance
importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# TOP FEATURES
# ============================================================

st.subheader(
    f"Top Features: {selected_model}"
)

top_n = st.slider(
    "Number of features to display",
    min_value=5,
    max_value=30,
    value=10,
    step=5
)

top_features = importance_df.head(
    top_n
)


# ============================================================
# BAR CHART
# ============================================================

fig = px.bar(
    top_features.sort_values(
        "Importance",
        ascending=True
    ),
    x="Importance",
    y="Feature",
    orientation="h",
    title=(
        f"Top {top_n} Features "
        f"for {selected_model}"
    ),
    text="Importance"
)

fig.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

fig.update_layout(
    height=max(
        450,
        top_n * 40
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


st.divider()


# ============================================================
# FEATURE IMPORTANCE TABLE
# ============================================================

st.subheader(
    "Feature Importance Ranking"
)

display_df = importance_df.copy()

display_df[
    "Importance"
] = display_df[
    "Importance"
].round(6)

display_df.insert(
    0,
    "Rank",
    range(
        1,
        len(display_df) + 1
    )
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ============================================================
# TOP 5 FEATURES
# ============================================================

st.subheader(
    "Top 5 Most Important Features"
)

top_5 = importance_df.head(
    5
)

cols = st.columns(5)

for index, row in top_5.iterrows():

    with cols[index]:

        st.metric(
            label=row["Feature"],
            value=f"{row['Importance']:.4f}"
        )


st.divider()


# ============================================================
# MODEL-SPECIFIC EXPLANATION
# ============================================================

st.subheader(
    "How Importance is Calculated"
)

if selected_model == "Logistic Regression":

    st.write(
        "For Logistic Regression, feature importance "
        "is calculated using the absolute value of "
        "the model coefficients. Larger absolute "
        "coefficients indicate a stronger contribution "
        "to the model's decision."
    )

elif selected_model == "Random Forest":

    st.write(
        "For Random Forest, feature importance is "
        "calculated using the model's built-in "
        "feature importance values based on how "
        "features contribute to the decision trees."
    )


# ============================================================
# XAI NOTE
# ============================================================

st.divider()

st.subheader(
    "Why This Matters for Explainable AI"
)

st.write(
    "Feature importance provides a global view of "
    "which measurements the model relies on most "
    "across the dataset."
)

st.write(
    "Later, SHAP analysis will provide a more detailed "
    "explanation by showing how individual features "
    "push a prediction toward benign or malignant."
)