import os
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dataset Overview",
    page_icon="📊",
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

PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "bc_dataset",
    "data.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_PATH)

    return data


data = load_data()


# ============================================================
# TITLE
# ============================================================

st.title("📊 Dataset Overview")

st.write(
    "Overview of the Breast Cancer Wisconsin dataset "
    "used for model training and evaluation."
)


st.divider()


# ============================================================
# DATASET INFORMATION
# ============================================================

st.subheader("Dataset Information")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Samples",
    data.shape[0]
)

col2.metric(
    "Total Columns",
    data.shape[1]
)

col3.metric(
    "Benign Cases",
    (data["diagnosis"] == "B").sum()
)

col4.metric(
    "Malignant Cases",
    (data["diagnosis"] == "M").sum()
)


st.divider()


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

st.subheader("Diagnosis Distribution")

diagnosis_data = pd.DataFrame({
    "Diagnosis": ["Benign", "Malignant"],
    "Count": [
        (data["diagnosis"] == "B").sum(),
        (data["diagnosis"] == "M").sum()
    ]
})


col1, col2 = st.columns(2)


with col1:

    fig = px.bar(
        diagnosis_data,
        x="Diagnosis",
        y="Count",
        title="Benign vs Malignant Cases",
        text="Count"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    fig = px.pie(
        diagnosis_data,
        names="Diagnosis",
        values="Count",
        title="Class Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


st.divider()


# ============================================================
# MISSING VALUES
# ============================================================

st.subheader("Missing Values")

missing_values = data.isnull().sum()

missing_values = missing_values[
    missing_values > 0
]


if missing_values.empty:

    st.success(
        "✅ No missing values found in the dataset."
    )

else:

    st.dataframe(
        missing_values.rename(
            "Missing Values"
        ),
        use_container_width=True
    )


st.divider()


# ============================================================
# DATASET PREVIEW
# ============================================================

st.subheader("Dataset Preview")

st.dataframe(
    data.head(10),
    use_container_width=True
)


st.divider()


# ============================================================
# FEATURE STATISTICS
# ============================================================

st.subheader("Feature Statistics")

numeric_data = data.select_dtypes(
    include="number"
)

st.dataframe(
    numeric_data.describe().T,
    use_container_width=True
)