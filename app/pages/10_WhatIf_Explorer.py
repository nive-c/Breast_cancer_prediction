import os
import pickle
import json
import pandas as pd
import streamlit as st

import sys
# Ensure app directory is on python path so we can import whatif_tools
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)

from whatif_tools import (
    MODEL_NAME,
    run_gemma_whatif,
    check_invented_numbers,
    check_flip_claim,
    FEATURE_NAMES,
    FEATURE_MIN,
    FEATURE_MAX
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="What-If Explorer",
    page_icon="🔮",
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
# LOAD DATA / MODEL / SCALER (CACHED)
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
# PAGE TITLE & DISCLAIMER
# --------------------------------------------------

st.title("🔮 What-If Explorer (Gemma 4 Function Calling)")

st.write(
    "Explore counterfactual scenarios and test how changes to cell nuclei "
    "measurements affect real machine learning models. Gemma 4 interprets your "
    "natural language question, calls real Python model functions, and explains "
    "the output with zero hallucinated numbers."
)

st.info(
    "⚠️ **Educational tool only.** This analysis reflects the behavior of the "
    "trained machine learning model and is **not a clinical diagnosis**. It provides "
    "no medical advice or treatment recommendations."
)

st.divider()


# --------------------------------------------------
# MODEL & SAMPLE SELECTION
# --------------------------------------------------

col_m, col_s = st.columns([1, 1])

with col_m:
    selected_model_name = st.selectbox(
        "Choose the model",
        ["Logistic Regression", "Random Forest"]
    )

if selected_model_name == "Logistic Regression":
    model_file = "logistic_regression"
else:
    model_file = "random_forest"

model = load_model(model_file)

with col_s:
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
# CURRENT PREDICTION METRICS
# --------------------------------------------------

actual_value = data.iloc[sample_index]["diagnosis"]
actual_label = "Malignant" if actual_value == 1 else "Benign"

prediction_val = int(model.predict(sample_scaled)[0])
prediction_label = "Malignant" if prediction_val == 1 else "Benign"
malignant_prob = float(model.predict_proba(sample_scaled)[0][1])

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Prediction", prediction_label)

with col2:
    st.metric("Malignant Probability", f"{malignant_prob * 100:.2f}%")

with col3:
    st.metric("Actual Dataset Label", actual_label)

st.divider()


# --------------------------------------------------
# WHAT-IF QUESTION INPUT & EXAMPLES
# --------------------------------------------------

st.subheader("Ask a What-If Question")
st.write(
    "Type a scenario to see how the model responds, or click an example button below:"
)

# Initialize session state for user question and last result
if "whatif_question" not in st.session_state:
    st.session_state["whatif_question"] = "What if radius_se were 25% lower?"

if "whatif_result" not in st.session_state:
    st.session_state["whatif_result"] = None

example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    if st.button("Example 1: Lower radius_se by 25%"):
        st.session_state["whatif_question"] = "What if radius_se were 25% lower?"

with example_col2:
    if st.button("Example 2: Smallest change to concavity_mean to flip?"):
        st.session_state["whatif_question"] = "What is the smallest change to concavity_mean that flips the prediction?"

with example_col3:
    if st.button("Example 3: Increase radius_mean by 50%"):
        st.session_state["whatif_question"] = "What if radius_mean were 50% higher?"

user_question = st.text_input(
    "Your What-If Question:",
    value=st.session_state["whatif_question"],
    key="question_input"
)

run_button = st.button("Run What-If Analysis", type="primary")


# --------------------------------------------------
# EXECUTION VIA GEMMA 4 TOOL LOOP
# --------------------------------------------------

if run_button:
    if not user_question.strip():
        st.warning("Please type a what-if question before running the analysis.")
    elif "GEMINI_API_KEY" not in os.environ or not os.environ["GEMINI_API_KEY"]:
        st.error("GEMINI_API_KEY environment variable is not set. Please set GEMINI_API_KEY.")
    else:
        with st.spinner(f"Asking Gemma 4 ({MODEL_NAME}) and executing model functions..."):
            try:
                result = run_gemma_whatif(
                    question=user_question,
                    sample_index=sample_index,
                    model_name=selected_model_name,
                    max_rounds=5
                )
                result["question"] = user_question
                result["sample_index"] = sample_index
                result["model_name"] = selected_model_name
                st.session_state["whatif_result"] = result
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")


# --------------------------------------------------
# DISPLAY RESULTS & AUTOMATIC CHECKS
# --------------------------------------------------

saved_result = st.session_state.get("whatif_result")

if saved_result:
    st.divider()
    st.subheader("Gemma 4 Explanation")
    st.write(saved_result["answer"])

    # Expandable tool call inspector
    with st.expander("🛠️ Tool calls Gemma made (Inspect real function execution)"):
        st.caption(
            f"Execution summary: {saved_result['rounds']} conversational round(s) in "
            f"{saved_result['elapsed_seconds']}s using model `{MODEL_NAME}`."
        )
        if saved_result["tool_calls"]:
            for i, tc in enumerate(saved_result["tool_calls"]):
                st.markdown(f"**Call {i+1}: `{tc['function']}`**")
                st.json({
                    "Arguments": tc["arguments"],
                    "Function Result": tc["result"]
                })
        else:
            st.info("No tool calls were invoked for this query.")

    # --------------------------------------------------
    # PART 4: AUTOMATIC INTEGRITY CHECKS
    # --------------------------------------------------
    st.divider()
    st.subheader("🔍 Automatic Integrity Checks")

    # 1. Invented-number check
    num_check = check_invented_numbers(
        saved_result["answer"],
        saved_result["question"],
        saved_result["tool_calls"]
    )

    if num_check["all_verified"]:
        st.success(
            f"✅ **Invented-Number Check Passed**: All {num_check['total_numbers']} number(s) "
            f"in Gemma's answer were traced to the real tool execution results or user question."
        )
    else:
        st.warning(
            f"⚠️ **Invented-Number Warning**: Detected numbers not directly traced to tool results: "
            f"{num_check['unsupported']}."
        )

    # 2. Flip-claim check
    flip_check = check_flip_claim(
        saved_result["answer"],
        saved_result["tool_calls"]
    )

    if flip_check["status"] == "match":
        st.success(f"✅ **Flip-Claim Check Passed**: {flip_check['detail']}")
    elif flip_check["status"] == "mismatch":
        st.error(f"❌ **Flip-Claim Check Mismatch**: {flip_check['detail']}")
    else:
        st.info(f"ℹ️ **Flip-Claim Check**: {flip_check['detail']}")

    st.caption(
        "**What these checks prove:** They verify that every number reported by Gemma matches "
        "the actual output returned by the scikit-learn model, and that statements about classification "
        "flips align with computed labels. They do not prove clinical validity or biological causation."
    )
