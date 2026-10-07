import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(BASE_DIR, "..", "..", "Model", "metrics.json")


def load_metrics():
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def build_metrics_df(metrics):
    rows = []
    for key, m in metrics.items():
        rows.append({
            "key": key,
            "Model": m["display_name"],
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1_score"],
            "ROC-AUC": m["roc_auc"],
        })
    return pd.DataFrame(rows)


def get_best_model_key(metrics):
    # best model is picked by F1-score, matching how the prediction page selects a model
    return max(metrics, key=lambda k: metrics[k]["f1_score"])


def main():
    st.set_page_config(
        page_title="Model Comparison",
        page_icon=":bar_chart:",
        layout="wide"
    )

    st.title("Model Comparison (Internal Test)")
    st.write(
        "Performance of all models on the internal test partition, "
        "held out from training."
    )

    metrics = load_metrics()
    df = build_metrics_df(metrics)
    best_key = get_best_model_key(metrics)
    best_row = df[df["key"] == best_key].iloc[0]

    # --- metrics table ---
    display_df = df.drop(columns=["key"]).set_index("Model")

    def highlight_best(row):
        is_best = row.name == best_row["Model"]
        return ["background-color: #e6f4ea; font-weight: bold" if is_best else "" for _ in row]

    st.subheader("Metrics Table")
    st.dataframe(
        display_df.style.apply(highlight_best, axis=1).format("{:.4f}"),
        use_container_width=True
    )

    # --- bar chart ---
    st.subheader("Performance Comparison (Bar Chart)")
    metric_choice = st.selectbox(
        "Metric to compare",
        ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        index=0
    )

    colors = ["#4C78A8", "#F58518", "#54A24B"]
    fig = go.Figure(data=[
        go.Bar(
            x=df["Model"],
            y=df[metric_choice],
            marker_color=colors[:len(df)],
            text=df[metric_choice].apply(lambda v: f"{v:.4f}"),
            textposition="outside",
        )
    ])
    fig.update_layout(
        yaxis_title=metric_choice,
        yaxis_range=[0, 1.05],
        showlegend=False,
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- best model callout ---
    st.subheader("Best Model")
    st.success(
        f"**{best_row['Model']}** — selected as the best model "
        f"(highest F1-score: {best_row['F1-Score']:.4f})\n\n"
        f"Accuracy: {best_row['Accuracy']:.4f} · "
        f"Precision: {best_row['Precision']:.4f} · "
        f"Recall: {best_row['Recall']:.4f} · "
        f"ROC-AUC: {best_row['ROC-AUC']:.4f}"
    )
    st.caption(
        "Best model is chosen by F1-score rather than accuracy, since F1 balances "
        "precision and recall — important for a diagnostic task where both false "
        "positives and false negatives carry real cost."
    )


if __name__ == "__main__":
    main()