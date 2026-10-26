import io
import os
import pickle
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from PIL import Image

from gemma_explainer import top_features, ask_gemma, parse_reply, faithfulness


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Explain with Gemma",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(PROJECT_DIR, "bc_dataset", "data.csv")
MODEL_DIR = os.path.join(PROJECT_DIR, "Model", "models")
SCALER_PATH = os.path.join(PROJECT_DIR, "Model", "scaler.pkl")


# --------------------------------------------------
# LOAD DATA / MODEL / SCALER
# --------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(["id", "Unnamed: 32"], axis=1)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data


@st.cache_resource
def load_model(model_name):
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)
    return scaler


data = load_data()
scaler = load_scaler()

X = data.drop("diagnosis", axis=1)
X_scaled = scaler.transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)


# --------------------------------------------------
# PROMPT BUILDER
# --------------------------------------------------

def build_prompt(prediction, probability, model_name, feature_names, feature_values, shap_vals):
    top = top_features(feature_names, shap_vals, 5)

    lines = ""
    for name, value in top:
        idx = list(feature_names).index(name)
        if value > 0:
            direction = "towards malignant"
        else:
            direction = "towards benign"
        lines += f"- {name}: raw value={feature_values[idx]:.3f}, shap={value:.4f} ({direction})\n"

    prompt = f"""You are explaining a machine learning model's output to a student.
Model: {model_name}. It predicted: {prediction} (malignant probability {probability * 100:.1f}%).
Top features by SHAP contribution:
{lines}
The attached image is the SHAP waterfall plot for the same sample. Its feature values are standardized, so they differ from the raw values listed above.

Rules:
- Only use the numbers above and the image. Do not invent values or features.
- Positive SHAP pushes towards malignant, negative towards benign.
- Do not say this is a medical diagnosis. It is only the model's prediction.
- Keep it under 120 words, in simple English.

Reply ONLY with JSON in this format:
{{"explanation": "...", "top_features": ["feature1", "feature2", "feature3"]}}
where top_features are the 3 features with the largest absolute SHAP value (ignore the sign, so a large negative value counts), from largest to smallest."""

    return prompt


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("🤖 Explain with Gemma 4")

st.write(
    "Gemma 4 reads the SHAP results for one sample and explains "
    "them in plain English. Then we check whether its explanation "
    "matches the real SHAP ranking."
)

st.info(
    "Educational tool only. This is the model's prediction, "
    "not a medical diagnosis."
)

st.divider()


# --------------------------------------------------
# MODEL + SAMPLE SELECTION
# --------------------------------------------------

selected_model = st.selectbox(
    "Choose the model",
    ["Logistic Regression", "Random Forest"]
)

if selected_model == "Logistic Regression":
    model_file = "logistic_regression"
else:
    model_file = "random_forest"

model = load_model(model_file)

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
# PREDICTION
# --------------------------------------------------

actual_value = data.iloc[sample_index]["diagnosis"]
if actual_value == 1:
    actual_label = "Malignant"
else:
    actual_label = "Benign"

prediction = model.predict(sample_scaled)[0]
probability = model.predict_proba(sample_scaled)[0][1]

if prediction == 1:
    prediction_label = "Malignant"
else:
    prediction_label = "Benign"

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Prediction", prediction_label)

with col2:
    st.metric("Malignant Probability", f"{probability * 100:.2f}%")

with col3:
    st.metric("Actual Dataset Label", actual_label)


# --------------------------------------------------
# SHAP
# --------------------------------------------------

st.divider()

st.subheader("SHAP explanation")

with st.spinner("Calculating SHAP explanation..."):

    if selected_model == "Logistic Regression":
        explainer = shap.LinearExplainer(model, X_scaled_df)
        shap_values = explainer(sample_scaled)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(sample_scaled)

# random forest can give values for both classes, keep the malignant one
if len(shap_values.values.shape) == 3:
    shap_values = shap_values[:, :, 1]

fig = plt.figure()

shap.plots.waterfall(shap_values[0], max_display=10, show=False)

# keep a copy of the plot to send to Gemma
buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight")
buf.seek(0)
waterfall_image = Image.open(buf)

st.pyplot(fig, clear_figure=True)
plt.close(fig)

feature_names = list(X.columns)
feature_values = sample.iloc[0].values
sv = shap_values.values[0]


# --------------------------------------------------
# GEMMA
# --------------------------------------------------

st.divider()

st.subheader("Gemma's explanation")

if st.button("Explain this prediction with Gemma"):

    with st.spinner("Asking Gemma..."):
        prompt = build_prompt(
            prediction_label,
            probability,
            selected_model,
            feature_names,
            feature_values,
            sv
        )
        reply = ask_gemma(prompt, waterfall_image)

    data_reply = parse_reply(reply)

    if data_reply is None:
        st.error("Gemma's reply was not valid JSON. Try again.")
        st.write(reply)
    else:
        st.write(data_reply["explanation"])

        real_top = [name for name, value in top_features(feature_names, sv, 3)]
        score = faithfulness(real_top, data_reply["top_features"], 3)

        st.write("Real SHAP top 3:", real_top)
        st.write("Gemma's top 3:", data_reply["top_features"])

        if score == 1.0:
            st.success("Verified: Gemma's top features match SHAP")
        else:
            st.warning(f"Mismatch: only {int(round(score * 3))}/3 features match SHAP")