from pathlib import Path

import joblib
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "naive_bayes": "Gaussian Naive Bayes",
    "svm": "Support Vector Machine",
}


@st.cache_resource
def load_model(model_name):
    return joblib.load(MODEL_DIR / f"{model_name}.joblib")


@st.cache_data
def load_saved_metrics():
    return pd.read_csv(PROJECT_DIR / "model_metrics.csv", index_col="model")


st.set_page_config(page_title="Bank Marketing Classification", layout="wide")
st.title("Bank Marketing Classification")
st.write("Evaluate five machine-learning classifiers on the UCI Bank Marketing dataset.")

tab1, tab2 = st.tabs(["Upload and evaluate", "Model comparison"])

with tab2:
    if (PROJECT_DIR / "model_metrics.csv").exists():
        metrics = load_saved_metrics().copy()
        metrics.index = [MODEL_LABELS.get(name, name) for name in metrics.index]
        st.subheader("Saved test-set metrics")
        st.dataframe(metrics.style.format("{:.4f}"), use_container_width=True)
    else:
        st.warning("Run `python train_models.py` to generate the model metrics.")

with tab1:
    selected_label = st.selectbox("Select model", list(MODEL_LABELS.values()))
    model_name = next(name for name, label in MODEL_LABELS.items() if label == selected_label)
    model_path = MODEL_DIR / f"{model_name}.joblib"

    if not model_path.exists():
        st.error("Models have not been trained. Run `python train_models.py` first.")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload bank test CSV with customer features and a `y` column",
        type=["csv"],
    )
    default_test_path = PROJECT_DIR / "test_data.csv"
    use_default = st.checkbox("Use bundled test data", value=uploaded_file is None)

    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
    elif use_default and default_test_path.exists():
        test_data = pd.read_csv(default_test_path)
    else:
        test_data = None

    if test_data is not None:
        if "y" not in test_data.columns:
            st.error("The CSV must contain a `y` column.")
        else:
            y_true_text = test_data["y"]
            X_test = test_data.drop(columns=["y"])
            valid_values = y_true_text.astype(str).isin(["0", "1", "no", "yes"])
            if not valid_values.all():
                st.error("`y` values must be 0/1 or no/yes.")
                st.stop()

            y_true = y_true_text.replace({"no": 0, "yes": 1, "0": 0, "1": 1})
            y_true = y_true.astype(int)

            model = load_model(model_name)
            try:
                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test)[:, 1]
            except ValueError as exc:
                st.error(f"The uploaded CSV does not match the trained feature schema: {exc}")
                st.stop()

            metrics = {
                "Accuracy": accuracy_score(y_true, predictions),
                "AUC": roc_auc_score(y_true, probabilities),
                "Precision": precision_score(y_true, predictions, zero_division=0),
                "Recall": recall_score(y_true, predictions, zero_division=0),
                "F1 Score": f1_score(y_true, predictions, zero_division=0),
                "MCC": matthews_corrcoef(y_true, predictions),
            }
            st.subheader("Evaluation metrics")
            metric_columns = st.columns(3)
            for column, (metric, value) in zip(metric_columns * 2, metrics.items()):
                column.metric(metric, f"{value:.4f}")

            cm = confusion_matrix(y_true, predictions, labels=[0, 1])
            left, right = st.columns(2)
            with left:
                st.subheader("Confusion matrix")
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["No subscription", "Subscription"],
                            yticklabels=["No subscription", "Subscription"], ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
            with right:
                st.subheader("Classification report")
                report = classification_report(
                    y_true, predictions,
                    target_names=["No subscription", "Subscription"],
                    output_dict=True,
                    zero_division=0,
                )
                st.dataframe(pd.DataFrame(report).transpose().style.format("{:.4f}"))
