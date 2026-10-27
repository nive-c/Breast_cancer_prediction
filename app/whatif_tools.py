import os
import re
import math
import time
import pickle
import pandas as pd
import numpy as np
from google import genai
from google.genai import types

# --------------------------------------------------
# CONFIGURATION & PATHS
# --------------------------------------------------

MODEL_NAME = "gemma-4-26b-a4b-it"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(PROJECT_DIR, "bc_dataset", "data.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "Model", "models")
SCALER_PATH = os.path.join(PROJECT_DIR, "Model", "scaler.pkl")


# --------------------------------------------------
# DATA & SCALER LOADING
# --------------------------------------------------

def load_clean_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(["id", "Unnamed: 32"], axis=1)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data


def load_model(model_name: str):
    # model_name: 'logistic_regression' or 'random_forest'
    path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


DATA = load_clean_data()
X_DATA = DATA.drop("diagnosis", axis=1)
FEATURE_NAMES = list(X_DATA.columns)
FEATURE_MIN = X_DATA.min().to_dict()
FEATURE_MAX = X_DATA.max().to_dict()
FEATURE_MEAN = X_DATA.mean().to_dict()

SCALER = load_scaler()
MODELS = {
    "Logistic Regression": load_model("logistic_regression"),
    "Random Forest": load_model("random_forest")
}

ACTIVE_MODEL_KEY = "Logistic Regression"


def set_active_model_key(key: str):
    global ACTIVE_MODEL_KEY
    if key in MODELS:
        ACTIVE_MODEL_KEY = key


def get_active_model():
    return MODELS[ACTIVE_MODEL_KEY]


# --------------------------------------------------
# PART 1: PURE PYTHON TOOLS
# --------------------------------------------------

def get_prediction(sample_index: int, model=None, scaler=None):
    # Original raw feature values -> scale with scaler -> predict & predict_proba
    m = model if model is not None else get_active_model()
    s = scaler if scaler is not None else SCALER

    if sample_index < 0 or sample_index >= len(X_DATA):
        raise ValueError(f"sample_index {sample_index} is out of range [0, {len(X_DATA)-1}].")

    sample_df = X_DATA.iloc[[sample_index]]
    scaled_arr = s.transform(sample_df)

    pred = int(m.predict(scaled_arr)[0])
    prob = float(m.predict_proba(scaled_arr)[0][1])
    label = "Malignant" if pred == 1 else "Benign"

    return {
        "sample_index": int(sample_index),
        "prediction": label,
        "malignant_probability": round(prob, 4)
    }


def simulate_change(sample_index: int, feature: str, percent_change: float, model=None, scaler=None):
    # Copy sample, multiply that ONE feature by (1 + percent_change/100), clamp to min/max, predict
    m = model if model is not None else get_active_model()
    s = scaler if scaler is not None else SCALER

    if feature not in FEATURE_NAMES:
        raise ValueError(f"Unknown feature '{feature}'. Must be one of the 30 real feature names: {FEATURE_NAMES}")

    if sample_index < 0 or sample_index >= len(X_DATA):
        raise ValueError(f"sample_index {sample_index} is out of range [0, {len(X_DATA)-1}].")

    sample_df = X_DATA.iloc[[sample_index]].copy()
    orig_val = float(sample_df[feature].values[0])

    unclamped_val = orig_val * (1.0 + float(percent_change) / 100.0)
    min_val = float(FEATURE_MIN[feature])
    max_val = float(FEATURE_MAX[feature])

    clamped = False
    if unclamped_val < min_val:
        new_val = min_val
        clamped = True
    elif unclamped_val > max_val:
        new_val = max_val
        clamped = True
    else:
        new_val = unclamped_val

    # Old prediction
    old_scaled = s.transform(sample_df)
    old_pred = int(m.predict(old_scaled)[0])
    old_prob = float(m.predict_proba(old_scaled)[0][1])
    old_label = "Malignant" if old_pred == 1 else "Benign"

    # New prediction
    sample_df[feature] = new_val
    new_scaled = s.transform(sample_df)
    new_pred = int(m.predict(new_scaled)[0])
    new_prob = float(m.predict_proba(new_scaled)[0][1])
    new_label = "Malignant" if new_pred == 1 else "Benign"

    flipped = bool(old_pred != new_pred)

    return {
        "feature": feature,
        "percent_change": float(percent_change),
        "original_value": round(orig_val, 4),
        "new_value": round(new_val, 4),
        "clamped": clamped,
        "old_malignant_probability": round(old_prob, 4),
        "new_malignant_probability": round(new_prob, 4),
        "old_label": old_label,
        "new_label": new_label,
        "flipped": flipped
    }


def find_flip_change(sample_index: int, feature: str, model=None, scaler=None):
    # Search percent changes from -90 to +300 in steps within clamped dataset range
    # Return smallest absolute percent change that flips the label, or "no flip within range"
    m = model if model is not None else get_active_model()
    s = scaler if scaler is not None else SCALER

    if feature not in FEATURE_NAMES:
        raise ValueError(f"Unknown feature '{feature}'. Must be one of the 30 real feature names: {FEATURE_NAMES}")

    if sample_index < 0 or sample_index >= len(X_DATA):
        raise ValueError(f"sample_index {sample_index} is out of range [0, {len(X_DATA)-1}].")

    orig_pred_info = get_prediction(sample_index, model=m, scaler=s)
    orig_label = orig_pred_info["prediction"]
    orig_prob = orig_pred_info["malignant_probability"]
    orig_pred = 1 if orig_label == "Malignant" else 0
    orig_val = float(X_DATA.iloc[sample_index][feature])
    min_val = float(FEATURE_MIN[feature])
    max_val = float(FEATURE_MAX[feature])

    # Search candidates ordered by absolute percent change: 1, -1, 2, -2, ..., 90, -90, 91, ..., 300
    candidates = []
    for step in range(1, 301):
        if step <= 300:
            candidates.append(step)
        if step <= 90:
            candidates.append(-step)

    sample_row = X_DATA.iloc[sample_index]
    rows = []
    valid_candidates = []
    for pct in candidates:
        unclamped = orig_val * (1.0 + pct / 100.0)
        if unclamped < min_val:
            clamped_val = min_val
            is_clamped = True
        elif unclamped > max_val:
            clamped_val = max_val
            is_clamped = True
        else:
            clamped_val = unclamped
            is_clamped = False

        r = sample_row.copy()
        r[feature] = clamped_val
        rows.append(r)
        valid_candidates.append((pct, clamped_val, is_clamped))

    # Fast batch evaluation
    df_batch = pd.DataFrame(rows)
    scaled_batch = s.transform(df_batch)
    preds = m.predict(scaled_batch)
    probas = m.predict_proba(scaled_batch)[:, 1]

    for i, pred in enumerate(preds):
        if int(pred) != orig_pred:
            pct, new_v, is_cl = valid_candidates[i]
            return {
                "feature": feature,
                "smallest_percent_change": pct,
                "original_value": round(orig_val, 4),
                "new_value": round(new_v, 4),
                "clamped": is_cl,
                "old_malignant_probability": round(orig_prob, 4),
                "new_malignant_probability": round(float(probas[i]), 4),
                "old_label": orig_label,
                "new_label": "Malignant" if int(pred) == 1 else "Benign",
                "flipped": True
            }

    return {
        "feature": feature,
        "result": "no flip within range",
        "search_range": "-90% to +300%",
        "original_value": round(orig_val, 4),
        "malignant_probability": round(orig_prob, 4),
        "label": orig_label,
        "flipped": False
    }


# Map of available tools for dynamic invocation
TOOL_FUNCTIONS = {
    "get_prediction": get_prediction,
    "simulate_change": simulate_change,
    "find_flip_change": find_flip_change
}


# --------------------------------------------------
# PART 2: GEMMA 4 FUNCTION CALLING & SCHEMAS
# --------------------------------------------------

get_prediction_declaration = types.FunctionDeclaration(
    name="get_prediction",
    description="Gets the machine learning model's prediction and malignant probability for a given dataset sample index.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "sample_index": types.Schema(
                type=types.Type.INTEGER,
                description="The zero-based integer index of the patient sample in the dataset."
            )
        },
        required=["sample_index"]
    )
)

simulate_change_declaration = types.FunctionDeclaration(
    name="simulate_change",
    description="Simulates changing a single feature by a specified percentage for a given sample, clamped to the dataset min/max range, and calculates the new model prediction and malignant probability.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "sample_index": types.Schema(
                type=types.Type.INTEGER,
                description="The zero-based integer index of the patient sample."
            ),
            "feature": types.Schema(
                type=types.Type.STRING,
                description="The exact feature name to modify, must be one of the 30 real feature names (e.g. 'radius_se', 'concavity_mean')."
            ),
            "percent_change": types.Schema(
                type=types.Type.NUMBER,
                description="The percentage to change the feature by (e.g. -25 for 25% lower, 50 for 50% higher)."
            )
        },
        required=["sample_index", "feature", "percent_change"]
    )
)

find_flip_change_declaration = types.FunctionDeclaration(
    name="find_flip_change",
    description="Finds the smallest absolute percentage change (from -90% to +300%) in a single feature that flips the model prediction between Benign and Malignant, or indicates no flip within range.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "sample_index": types.Schema(
                type=types.Type.INTEGER,
                description="The zero-based integer index of the patient sample."
            ),
            "feature": types.Schema(
                type=types.Type.STRING,
                description="The exact feature name to search for a prediction flip, must be one of the 30 real feature names."
            )
        },
        required=["sample_index", "feature"]
    )
)

GEMMA_TOOLS = [
    types.Tool(function_declarations=[
        get_prediction_declaration,
        simulate_change_declaration,
        find_flip_change_declaration
    ])
]

SYSTEM_INSTRUCTION = (
    "You are an AI assistant explaining machine learning model behavior on breast cancer cell nuclei measurements.\n"
    "Rules:\n"
    "1. Explain ONLY using values returned by the provided functions. Never state or invent numbers that did not come from a function result or the user's prompt.\n"
    "2. This describes the MODEL'S behavior only, NOT a clinical or medical diagnosis.\n"
    "3. Give NO clinical advice (no mention of biopsy, treatment, or second opinions).\n"
    "4. If the question cannot be answered with the tools, clearly say so.\n"
)


def get_genai_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set GEMINI_API_KEY.")
    return genai.Client(api_key=api_key)


def run_gemma_whatif(question: str, sample_index: int, model_name: str, max_rounds: int = 5):
    # Runs the manual tool loop: sends question, executes function calls, returns final answer
    client = get_genai_client()

    # Set the active model key so tool calls use the user's selected model
    set_active_model_key(model_name)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=GEMMA_TOOLS,
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL")
    )

    # Contextual prompt ensuring Gemma knows the active sample and model
    full_user_prompt = (
        f"The user is inspecting Sample {sample_index} with the {model_name} model.\n"
        f"User question: {question}"
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=full_user_prompt)]
        )
    ]

    tool_calls_log = []
    start_time = time.time()

    for round_num in range(max_rounds):
        # Call API with retry on 500 error
        res = None
        for attempt in range(2):
            try:
                res = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,
                    config=config
                )
                break
            except Exception as e:
                if attempt == 0:
                    time.sleep(1)
                else:
                    raise

        candidate = res.candidates[0]
        contents.append(candidate.content)

        # Check for function calls
        function_calls = [p.function_call for p in candidate.content.parts if p.function_call]
        if not function_calls:
            # Gemma finished and returned text answer
            elapsed = time.time() - start_time
            return {
                "answer": res.text,
                "tool_calls": tool_calls_log,
                "rounds": round_num + 1,
                "elapsed_seconds": round(elapsed, 2)
            }

        # Execute each function call
        tool_responses = []
        for fc in function_calls:
            fn_name = fc.name
            fn_args = dict(fc.args) if fc.args else {}

            # Fallback if sample_index omitted by model
            if "sample_index" not in fn_args:
                fn_args["sample_index"] = sample_index

            if fn_name in TOOL_FUNCTIONS:
                try:
                    result = TOOL_FUNCTIONS[fn_name](**fn_args)
                except Exception as err:
                    result = {"error": str(err)}
            else:
                result = {"error": f"Unknown function '{fn_name}'"}

            tool_calls_log.append({
                "function": fn_name,
                "arguments": fn_args,
                "result": result
            })

            tool_responses.append(
                types.Part.from_function_response(
                    name=fn_name,
                    response={"result": result}
                )
            )

        contents.append(
            types.Content(
                role="tool",
                parts=tool_responses
            )
        )

    elapsed = time.time() - start_time
    return {
        "answer": "Reached maximum tool rounds without final answer.",
        "tool_calls": tool_calls_log,
        "rounds": max_rounds,
        "elapsed_seconds": round(elapsed, 2)
    }


# --------------------------------------------------
# PART 4: AUTOMATIC VERIFICATION CHECKS
# --------------------------------------------------

def extract_numbers_from_text(text: str):
    # Extracts floating point and integer numbers from a text string
    raw_matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)
    numbers = []
    for m in raw_matches:
        try:
            numbers.append(float(m))
        except ValueError:
            pass
    return numbers


def gather_allowed_numbers(question: str, tool_calls: list):
    allowed = set()

    # Numbers in user question
    for n in extract_numbers_from_text(question):
        allowed.add(n)

    # Numbers in tool calls arguments & results
    def harvest(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                harvest(v)
        elif isinstance(obj, list):
            for item in obj:
                harvest(item)
        elif isinstance(obj, (int, float)):
            val = float(obj)
            allowed.add(val)
            # Add percentage equivalents for probabilities (e.g. 1.0 -> 100, 0.05 -> 5)
            if 0.0 <= val <= 1.0:
                allowed.add(round(val * 100.0, 2))
                allowed.add(round(val * 100.0, 1))
                allowed.add(round(val * 100.0, 0))
        elif isinstance(obj, str):
            for n in extract_numbers_from_text(obj):
                allowed.add(n)

    for tc in tool_calls:
        harvest(tc.get("arguments", {}))
        harvest(tc.get("result", {}))

    return allowed


def check_invented_numbers(answer_text: str, question: str, tool_calls: list):
    answer_numbers = extract_numbers_from_text(answer_text)
    allowed_numbers = gather_allowed_numbers(question, tool_calls)

    supported = []
    unsupported = []

    for n in answer_numbers:
        matched = False
        for c in allowed_numbers:
            # Check close match or percentage scale match
            if math.isclose(n, c, abs_tol=0.015) or math.isclose(n, c * 100.0, abs_tol=0.5):
                matched = True
                break
            # Check absolute value match (e.g. -25 vs 25)
            if math.isclose(abs(n), abs(c), abs_tol=0.015):
                matched = True
                break

        if matched:
            supported.append(n)
        else:
            unsupported.append(n)

    return {
        "all_verified": len(unsupported) == 0,
        "supported": supported,
        "unsupported": unsupported,
        "total_numbers": len(answer_numbers)
    }


def check_flip_claim(answer_text: str, tool_calls: list):
    tool_flipped_values = []
    for tc in tool_calls:
        res = tc.get("result", {})
        if "flipped" in res:
            tool_flipped_values.append(res["flipped"])

    if not tool_flipped_values:
        return {
            "status": "inconclusive",
            "detail": "No tool returned a flip result."
        }

    tool_flipped = any(tool_flipped_values)
    text_lower = answer_text.lower()

    no_flip_indicators = [
        "does not flip", "no flip", "would not flip", "remains", "would remain",
        "no change", "cannot flip", "will remain", "stayed", "did not flip"
    ]
    flip_indicators = [
        "flips the prediction", "would flip", "flips from", "will flip",
        "changed from", "changes to", "flips to", "flipped"
    ]

    claims_no_flip = any(ind in text_lower for ind in no_flip_indicators)
    claims_flip = any(ind in text_lower for ind in flip_indicators)

    if claims_no_flip and not claims_flip:
        claimed_flipped = False
    elif claims_flip and not claims_no_flip:
        claimed_flipped = True
    else:
        if "no" in text_lower and "flip" in text_lower:
            claimed_flipped = False
        elif "flip" in text_lower:
            claimed_flipped = True
        else:
            return {
                "status": "inconclusive",
                "detail": "No explicit flip claim detected in the text."
            }

    if claimed_flipped == tool_flipped:
        return {
            "status": "match",
            "tool_flipped": tool_flipped,
            "claimed_flipped": claimed_flipped,
            "detail": f"Claim verified: Model {'flips' if tool_flipped else 'does not flip'}, which matches Gemma's statement."
        }
    else:
        return {
            "status": "mismatch",
            "tool_flipped": tool_flipped,
            "claimed_flipped": claimed_flipped,
            "detail": f"Claim mismatch: Tool reported flipped={tool_flipped}, but Gemma text claimed flipped={claimed_flipped}."
        }
