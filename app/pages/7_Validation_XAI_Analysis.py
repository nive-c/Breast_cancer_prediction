import os
import sys
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

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from gemma_explainer import (
    top_features,
    ask_gemma,
    parse_reply,
    faithfulness,
    build_error_prompt,
    build_prompt
)

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
error_indices = [i for i, is_err in enumerate(validation_errors) if is_err]

error_count = len(error_indices)
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
        f"The model made **{error_count} incorrect prediction(s)** on the "
        "simulated external validation set. Explaining why the model got a sample "
        "wrong is often more interesting and valuable than explaining a correct prediction, "
        "as it exposes which features misled the model."
    )

    error_rows = []
    for idx in error_indices:
        act = "Malignant" if y_validation.iloc[idx] == 1 else "Benign"
        pred = "Malignant" if y_validation_pred[idx] == 1 else "Benign"
        err_type = "False Negative" if y_validation.iloc[idx] == 1 else "False Positive"
        prob = y_validation_proba[idx]
        error_rows.append({
            "Validation Sample Index": idx,
            "Actual Diagnosis": act,
            "Model Prediction": pred,
            "Error Type": err_type,
            "Malignant Probability": f"{prob:.2%}"
        })
    error_summary_df = pd.DataFrame(error_rows)
    st.dataframe(
        error_summary_df,
        use_container_width=True,
        hide_index=True
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

top_global_features = shap_importance.head(5)["Feature"].tolist()

feature_text = ", ".join(top_global_features)

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
# INDIVIDUAL VALIDATION CASE & ERROR EXPLANATION
# --------------------------------------------------

st.subheader("🔬 Explain an Individual Validation Prediction")

st.write(
    "Inspect the SHAP feature contributions for a specific validation sample. "
    "Explaining why the model got an error wrong is often more insightful "
    "than explaining a correct prediction."
)

if error_count > 0:
    sample_mode = st.radio(
        "Choose sample selection mode:",
        ["Misclassified Samples (Errors)", "All Validation Samples"],
        horizontal=True,
        help="Focusing on errors explains the specific features that misled the model."
    )

    if sample_mode == "Misclassified Samples (Errors)":
        error_label_map = {
            f"Sample #{idx} | Actual: {'Malignant' if y_validation.iloc[idx] == 1 else 'Benign'}, Predicted: {'Malignant' if y_validation_pred[idx] == 1 else 'Benign'} ({'False Negative' if y_validation.iloc[idx] == 1 else 'False Positive'}, Prob: {y_validation_proba[idx]:.1%})": idx
            for idx in error_indices
        }
        selected_label = st.selectbox(
            "Select an Error Sample to Explain",
            list(error_label_map.keys())
        )
        sample_index = error_label_map[selected_label]
    else:
        sample_index = st.number_input(
            "Validation Sample Index",
            min_value=0,
            max_value=len(X_validation) - 1,
            value=0,
            step=1
        )
else:
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
is_error = sample_prediction != sample_actual

actual_label = "Malignant" if sample_actual == 1 else "Benign"
prediction_label = "Malignant" if sample_prediction == 1 else "Benign"

col1, col2, col3 = st.columns(3)

col1.metric(
    "Actual Diagnosis",
    actual_label
)

col2.metric(
    "Model Prediction",
    prediction_label
)

col3.metric(
    "Malignant Probability",
    f"{sample_probability:.2%}"
)

if is_error:
    if sample_actual == 1:
        st.error(
            "⚠️ **Prediction Error: False Negative** — The true diagnosis is **Malignant**, "
            "but the model incorrectly predicted **Benign**."
        )
    else:
        st.error(
            "⚠️ **Prediction Error: False Positive** — The true diagnosis is **Benign**, "
            "but the model incorrectly predicted **Malignant**."
        )
else:
    st.success(
        "✅ **Correct Prediction**: The model's prediction matches the true diagnosis."
    )


# --------------------------------------------------
# SAMPLE SHAP VALUES
# --------------------------------------------------

if selected_model_name == "SVM" and sample_index >= len(shap_array):
    with st.spinner("Computing SHAP values for this validation sample..."):
        svm_sample_shap = explainer.shap_values(
            X_validation_scaled[sample_index:sample_index + 1]
        )
        if isinstance(svm_sample_shap, list):
            sample_shap = svm_sample_shap[1][0]
        else:
            sample_shap = svm_sample_shap[0]
else:
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

if is_error:
    st.write(
        f"The model incorrectly predicted **{prediction_label.lower()}** instead of **{actual_label.lower()}**."
    )
    if sample_prediction == 0:  # False Negative
        if not top_negative.empty:
            neg_feats = ", ".join(top_negative["Feature"].tolist())
            st.write(
                f"**Misleading features** pushing the prediction toward benign: **{neg_feats}**."
            )
        if not top_positive.empty:
            pos_feats = ", ".join(top_positive["Feature"].tolist())
            st.write(
                f"**Counteracting features** pushing toward malignant (outweighed): **{pos_feats}**."
            )
    else:  # False Positive
        if not top_positive.empty:
            pos_feats = ", ".join(top_positive["Feature"].tolist())
            st.write(
                f"**Misleading features** pushing the prediction toward malignant: **{pos_feats}**."
            )
        if not top_negative.empty:
            neg_feats = ", ".join(top_negative["Feature"].tolist())
            st.write(
                f"**Counteracting features** pushing toward benign (outweighed): **{neg_feats}**."
            )
else:
    prediction_direction = "malignant" if sample_prediction == 1 else "benign"
    st.write(
        f"The model predicted this validation sample as **{prediction_direction}**."
    )
    if not top_positive.empty:
        positive_features = ", ".join(top_positive["Feature"].tolist())
        st.write(
            f"Features pushing the prediction toward malignant include **{positive_features}**."
        )
    if not top_negative.empty:
        negative_features = ", ".join(top_negative["Feature"].tolist())
        st.write(
            f"Features pushing the prediction toward benign include **{negative_features}**."
        )


# --------------------------------------------------
# GROUNDED GEMMA EXPLANATION
# --------------------------------------------------

st.divider()

st.subheader("🤖 Grounded Explanation with Gemma")

st.write(
    "Gemma explains the model's decision in plain English. "
    "To keep the explanation grounded and truthful, Gemma receives "
    "**only the SHAP evidence and the labels** — no outside medical knowledge is assumed."
)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    api_key = st.text_input(
        "Enter GEMINI_API_KEY to enable Gemma:",
        type="password",
        help="You can also set the GEMINI_API_KEY environment variable."
    )

button_text = (
    "Explain this error with Gemma"
    if is_error
    else "Explain this prediction with Gemma"
)

feature_names = list(X_validation.columns)
feature_values = X_validation.iloc[sample_index].values

if st.button(button_text, type="primary"):
    if not api_key:
        st.error("Please provide a GEMINI_API_KEY to generate the explanation.")
    else:
        with st.spinner("Generating grounded Gemma explanation..."):
            try:
                if is_error:
                    prompt = build_error_prompt(
                        model_name=selected_model_name,
                        actual_label=actual_label,
                        prediction_label=prediction_label,
                        probability=sample_probability,
                        feature_names=feature_names,
                        feature_values=feature_values,
                        shap_vals=sample_shap
                    )
                else:
                    prompt = build_prompt(
                        prediction=prediction_label,
                        feature_names=feature_names,
                        feature_values=feature_values,
                        shap_values=sample_shap,
                        probability=sample_probability,
                        model_name=selected_model_name
                    )

                reply = ask_gemma(prompt, api_key=api_key)
                data_reply = parse_reply(reply)

                if data_reply is None:
                    st.error("Gemma's reply was not valid JSON. Response:")
                    st.write(reply)
                else:
                    st.markdown("### 📝 Gemma's Grounded Explanation")
                    st.info(data_reply.get("explanation", ""))

                    # Verification / Faithfulness check
                    real_top_tuples = top_features(feature_names, sample_shap, 3)
                    real_top = [name for name, _ in real_top_tuples]
                    gemma_top = data_reply.get("top_features", [])

                    score = faithfulness(real_top, gemma_top, 3)

                    st.markdown("#### 🔍 Grounding & Faithfulness Verification")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("**Real Top-3 SHAP Features:**")
                        for rank, (name, val) in enumerate(real_top_tuples, 1):
                            st.write(f"{rank}. `{name}` (SHAP: {val:+.4f})")
                    with col_b:
                        st.write("**Gemma's Cited Features:**")
                        for rank, name in enumerate(gemma_top, 1):
                            st.write(f"{rank}. `{name}`")

                    if score == 1.0:
                        st.success("✅ **Verified**: Gemma's cited features match the top SHAP features (3/3).")
                    elif score > 0:
                        st.info(f"ℹ️ **Partial Match**: {int(round(score * 3))}/3 cited features match the top SHAP ranking.")
                    else:
                        st.warning("⚠️ **Mismatch**: Gemma's cited features differed from the top 3 SHAP features.")

            except Exception as e:
                st.error(f"Error calling Gemma: {e}")


st.caption(
    "Important: SHAP and Gemma describe model behavior and feature influence only. "
    "They do not represent medical causation or clinical diagnoses."
)