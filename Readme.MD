# VerifyXAI: Breast Cancer Prediction with Explainable AI and Gemma 4

**Live demo:** https://breastcancerprediction-duy4hpw764ihy9esyzudmr.streamlit.app/

A Streamlit app that predicts whether a breast mass is benign or malignant from cell-nuclei measurements, explains each prediction with SHAP, and uses **Gemma 4** to explain and explore those predictions in plain English. Every Gemma answer is **automatically checked** against the real model output, so you can see when it agrees with the evidence and when it does not.

> **Educational project only.** The app describes how a machine learning model behaves on a public research dataset. It is not a medical device and gives no diagnosis or medical advice.

---

## What is new in this project

Most "LLM + ML" projects let the language model write fluent text and stop there. This one treats Gemma's output as something to test.

1. **Gemma explains SHAP results, then gets audited.** Gemma receives the prediction and the top SHAP contributions and writes a short explanation. The app compares the features Gemma names with the real SHAP ranking and shows Verified / Partial / Mismatch.
2. **What-If Explorer with Gemma 4 function calling.** The user asks a question such as "What if radius_se were 25% lower?". Gemma does not calculate anything. It calls Python functions that re-run the real trained model, then explains the returned numbers.
3. **Automatic integrity checks on every what-if answer.**
   - *Invented-number check:* every number in Gemma's answer must appear in the tool results or in the user's question.
   - *Flip-claim check:* if Gemma says the prediction flips (or does not), that claim is compared with the `flipped` value the model actually returned.
4. **Grounded explanations of misclassified samples.** On the validation page, Gemma explains why the model got a sample wrong, using only SHAP evidence and the labels.
5. **Speed finding.** Gemma 4 spends many hidden "thinking" tokens by default. In our tests a single explanation took 40 to 80 seconds. Setting `thinking_level="MINIMAL"` brought a call down to about 5 seconds (see Notes).

---

## How it works

```
 WDBC dataset (569 samples, 30 numeric features)
            |
            v
 Clean + encode labels (M=1, B=0) -> StandardScaler (fit on training data only)
            |
            v
 Train 3 classifiers: Logistic Regression | Random Forest | SVM
            |
            +--> Metrics, model comparison, internal test vs held-out validation
            |
            v
 SHAP: which features pushed a prediction toward malignant or benign
            |
            +--------------------------+
            |                          |
            v                          v
 Explain with Gemma             What-If Explorer
 (SHAP evidence -> JSON)        (question -> Gemma picks a function)
            |                          |
            v                          v
 Check: do Gemma's top           Python runs the REAL model
 features match SHAP?            -> result returned to Gemma
                                       |
                                       v
                                Gemma explains the result
                                       |
                                       v
                                Checks: invented numbers? flip claim correct?
```

### Step by step: the Gemma explanation page

1. You pick a model and a dataset sample. The app shows the model's prediction and probability.
2. SHAP values are computed for that sample (malignant class).
3. The top 5 SHAP features, their raw values and their direction are put into a prompt. The prompt tells Gemma to use only those numbers, not to invent features, and not to call anything a diagnosis.
4. Gemma replies in JSON: an explanation and the 3 features it says mattered most.
5. The app compares Gemma's 3 features with the real top 3 by absolute SHAP value and shows the result.

### Step by step: the What-If Explorer

1. You choose a model and a sample, then type a what-if question.
2. Gemma receives the question, rules (use only function results, no clinical advice), and descriptions of three functions:
   - `get_prediction(sample_index)`
   - `simulate_change(sample_index, feature, percent_change)`: changes one feature, clamps it to the dataset's min/max, scales it and re-predicts
   - `find_flip_change(sample_index, feature)`: searches from -90% to +300% for the smallest change that flips the label
3. Gemma answers with a function call. Python runs it against the real scikit-learn model and sends the result back.
4. Gemma writes the final answer from those numbers (usually 2 rounds, about 5 seconds).
5. The page shows the answer, an expander listing every tool call with its arguments and raw result, and the two integrity checks.

**Why it works:** a language model cannot run your model and is unreliable at arithmetic, so it only does what it is good at (understanding the question, choosing a function, phrasing the result). All numbers come from the trained model, and the checks catch it if it slips.

---

## Pages

| Page | What it does |
|---|---|
| Main (`app/main.py`) | Prediction simulator with 30 sliders and a radar chart |
| 1 Dataset Overview | Class balance, distributions, statistics |
| 2 Model Comparison | Accuracy, precision, recall, F1, ROC-AUC for the three models |
| 3 Feature Importance | Global feature importance |
| 4 SHAP Explainability | Global SHAP analysis |
| 5 Individual SHAP | Waterfall explanation for one sample |
| 6 Simulated External Validation | Performance on a held-out partition of the same dataset |
| 7 Validation XAI Analysis | Internal test vs held-out validation, error analysis, grounded Gemma explanation of errors |
| 9 Gemma Explanation | Gemma explains a SHAP result, with the faithfulness check |
| 10 What-If Explorer | Gemma 4 function calling with integrity checks |

Example from the What-If Explorer (Logistic Regression, sample 73, a malignant case the model calls benign): the smallest change to `radius_worst` that flips the prediction is +100% (16.57 to 33.14), giving a malignant probability of 0.5074. That is an extreme, barely-over-the-line input, so it says something about the model's decision boundary, not about real tumors.

---

## Tech stack

Python, Streamlit, scikit-learn, SHAP, pandas, Plotly, matplotlib, and the `google-genai` SDK calling **Gemma 4** (`gemma-4-26b-a4b-it`) through the Gemini API. Gemma 4 is an open-weights model; this project uses it as a hosted model rather than self-hosting it.

## Run it locally

```
py -m pip install -r requirements.txt
set GEMINI_API_KEY=your_key_here
py -m streamlit run app/main.py
```

Get a key from Google AI Studio. Keep it in an environment variable or Streamlit secrets, never in the code or the repo. Newer keys start with `AQ.`, which is expected.

## Repository layout

```
Breast_cancer_prediction/
  app/
    main.py                     landing page and simulator
    gemma_explainer.py          prompts, Gemma call, parsing, faithfulness check
    whatif_tools.py             model functions, tool schemas, tool loop, checks
    pages/                      Streamlit pages
  Model/                        training script, saved models, scaler, metrics
  bc_dataset/                   WDBC data.csv
  requirements.txt
```

---

## Limitations and honest notes

- **Not clinical.** The model predicts the dataset's label from a research dataset. It does not examine patient images or diagnose anyone.
- **Simulated external validation.** The held-out partition comes from the same WDBC dataset. It is not an independent hospital or population, so it is not true external validation.
- **Small data.** The evaluation sets have 114 samples each, so differences of one or two samples can change a metric noticeably. Differences between models should not be over-read.
- **Checks are limited.** The top-3 check compares feature names and ignores order and direction. The number check confirms a number appears in the tool results, not that it is attached to the right feature. Gemma could also choose the wrong function or feature, which is why the tool-call log is shown.
- **Model selection.** The best model is chosen by F1 on an evaluation set. Cross-validation would be a cleaner way to select it.
- **What-if inputs can be extreme.** Changes are clamped to the dataset's min and max, and the page reports when clamping happens.
- **Images.** Gemma 4 can accept the SHAP plot image. The image input is currently switched off in the explanation page (`USE_IMAGE = False`), so the explanation uses text evidence only.
- **Hosting.** `gemma-4-31b-it` returned intermittent 500 errors during testing, so the app uses `gemma-4-26b-a4b-it`.
- **Thinking setting.** `thinking_budget` is rejected by Gemma 4 (400 error). `thinking_level` accepts `minimal` and `high`. The speed gain from `minimal` was measured on a handful of calls, not benchmarked formally.

## Data

Wisconsin Diagnostic Breast Cancer (WDBC) dataset: 569 samples, 30 features computed from digitized fine-needle aspirate images (10 measurements, each as mean, standard error and worst).

## Author

Nivetha C, B.Tech Information Technology, Sri Shakthi Institute of Engineering and Technology, Coimbatore.