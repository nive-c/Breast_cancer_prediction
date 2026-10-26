import os
import json
from google import genai
import time
from google.genai import types

# check the exact Gemma 4 model name in Google AI Studio / the competition page
MODEL_NAME = "gemma-4-26b-a4b-it"
USE_IMAGE = False

def get_client():
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def top_features(feature_names, shap_values, k=5):
    # sort features by absolute shap value, biggest first
    pairs = list(zip(feature_names, shap_values))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return pairs[:k]


def build_prompt(prediction, feature_names, feature_values, shap_values):
    top = top_features(feature_names, shap_values, 5)
    lines = ""
    for name, sv in top:
        idx = list(feature_names).index(name)
        direction = "towards malignant" if sv > 0 else "towards benign"
        lines += f"- {name}: value={feature_values[idx]:.3f}, shap={sv:.3f} ({direction})\n"

    prompt = f"""You are explaining a machine learning model's output to a student.
The model predicted: {prediction}.
These are the top features and their SHAP contributions:
{lines}
Rules:
- Only use the numbers above. Do not invent values or features.
- Do not say this is a medical diagnosis. It is only the model's prediction.
- Keep it under 120 words, in simple English.

Reply ONLY with JSON in this format:
{{"explanation": "...", "top_features": ["feature1", "feature2", "feature3"]}}
where top_features are the 3 features you say mattered most, in order."""
    return prompt


import time

def ask_gemma(prompt, image=None):
    if not USE_IMAGE:
        image = None
    client = get_client()
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
    print("Token usage:", response.usage_metadata)
    return response.text

def parse_reply(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except Exception:
        return None


def faithfulness(real_top, gemma_top, k=3):
    # how many of the real top-k features did gemma also name
    real = set(real_top[:k])
    said = set(gemma_top[:k])
    return len(real & said) / k