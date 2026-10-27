import os
import json
import time
from google import genai
from google.genai import types

# check the exact Gemma 4 model name in Google AI Studio / the competition page
MODEL_NAME = "gemma-4-26b-a4b-it"
USE_IMAGE = False

def get_client(api_key=None):
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    return genai.Client(api_key=key)


def top_features(feature_names, shap_values, k=5):
    # sort features by absolute shap value, biggest first
    pairs = list(zip(feature_names, shap_values))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return pairs[:k]


def build_prompt(prediction, feature_names, feature_values, shap_values, probability=None, model_name=None):
    top = top_features(feature_names, shap_values, 5)
    lines = ""
    for name, sv in top:
        idx = list(feature_names).index(name)
        direction = "towards malignant" if sv > 0 else "towards benign"
        lines += f"- {name}: value={feature_values[idx]:.3f}, shap={sv:.3f} ({direction})\n"

    header = "You are explaining a machine learning model's output to a student.\n"
    if model_name:
        header += f"Model: {model_name}.\n"
    if probability is not None:
        header += f"The model predicted: {prediction} (malignant probability: {probability * 100:.1f}%).\n"
    else:
        header += f"The model predicted: {prediction}.\n"

    prompt = f"""{header}These are the top features and their SHAP contributions:
{lines}
Rules:
- Only use the numbers above. Do not invent values or features.
- Do not say this is a medical diagnosis. It is only the model's prediction.
- Keep it under 120 words, in simple English.

Reply ONLY with JSON in this format:
{{"explanation": "...", "top_features": ["feature1", "feature2", "feature3"]}}
where top_features are the 3 features you say mattered most, in order."""
    return prompt


def build_error_prompt(model_name, actual_label, prediction_label, probability, feature_names, feature_values, shap_vals, k=5):
    top = top_features(feature_names, shap_vals, k)

    lines = ""
    for name, value in top:
        idx = list(feature_names).index(name)
        direction = "towards malignant" if value > 0 else "towards benign"

        # Pushed toward the wrong prediction or toward the true label
        if (value > 0 and prediction_label == "Malignant") or (value < 0 and prediction_label == "Benign"):
            role = "pushed toward the INCORRECT prediction (misleading)"
        else:
            role = "pushed toward the TRUE diagnosis (counteracting)"

        lines += f"- {name}: raw value={feature_values[idx]:.3f}, shap={value:.4f} ({direction} — {role})\n"

    error_type = (
        "False Negative (Actual Malignant, Predicted Benign)"
        if (actual_label == "Malignant" and prediction_label == "Benign")
        else "False Positive (Actual Benign, Predicted Malignant)"
    )

    prompt = f"""You are explaining why a machine learning model made an INCORRECT prediction on a patient sample.
Model: {model_name}
Actual Diagnosis (Ground Truth): {actual_label}
Model Prediction: {prediction_label} (malignant probability: {probability * 100:.1f}%)
Error Type: {error_type}

Top features by SHAP contribution:
{lines}

Rules:
- Ground your explanation ONLY on the labels and SHAP evidence above. Do not invent outside medical facts, unlisted features, or speculative symptoms.
- Explain in simple English why the model made this error: which misleading features drove the model to the wrong prediction, and how they outweighed the features pointing to the true diagnosis.
- State clearly that this is an explanation of the model's error, NOT a clinical diagnosis.
- Keep the explanation under 120 words.

Reply ONLY with valid JSON in this format:
{{"explanation": "...", "top_features": ["feature1", "feature2", "feature3"]}}
where top_features are the 3 features with the largest absolute SHAP value from the list above, in order of influence."""

    return prompt


def ask_gemma(prompt, image=None, api_key=None):
    if not USE_IMAGE:
        image = None
    client = get_client(api_key=api_key)
    if client is None:
        raise ValueError("GEMINI_API_KEY is not set.")
    contents = [prompt]
    if image is not None:
        contents.append(image)

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL")
    )

    start = time.time()
    try:
        response = client.models.generate_content(
            model=MODEL_NAME, contents=contents, config=config
        )
    except Exception as e:
        # this also gives one retry after a random 500
        print("first try failed:", e)
        response = client.models.generate_content(
            model=MODEL_NAME, contents=contents
        )
    print("Gemma call took", round(time.time() - start, 1), "seconds")
    if hasattr(response, "usage_metadata"):
        print("Token usage:", response.usage_metadata)
    return response.text


def parse_reply(text):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(text[start_idx:end_idx + 1])
            except Exception:
                pass
        return None


def faithfulness(real_top, gemma_top, k=3):
    # how many of the real top-k features did gemma also name
    if not real_top or not gemma_top:
        return 0.0
    real = set(real_top[:k])
    said = set(gemma_top[:k])
    return len(real & said) / k
