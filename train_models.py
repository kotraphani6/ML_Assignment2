from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "bank_data" / "bank-full.csv"
MODEL_DIR = PROJECT_DIR / "model"

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=5000, random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "svm": SVC(kernel="rbf", probability=True, random_state=42),
}


def load_data():
    data = pd.read_csv(DATA_FILE, sep=";")
    data["y"] = data["y"].map({"yes": 1, "no": 0})
    return data


def model_pipeline(name, model, categorical_features, numeric_features):
    numeric_steps = [] if name in {"decision_tree", "naive_bayes"} else [("scaler", StandardScaler())]
    numeric_transformer = "passthrough" if not numeric_steps else Pipeline(numeric_steps)
    preprocessing = ColumnTransformer([
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ], sparse_threshold=0)
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    data = load_data()
    X = data.drop(columns=["y"])
    y = data["y"]
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    results = []
    for name, model in MODELS.items():
        pipeline = model_pipeline(name, model, categorical_features, numeric_features)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "auc": roc_auc_score(y_test, probabilities),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
            "mcc": matthews_corrcoef(y_test, predictions),
        })
        joblib.dump(pipeline, MODEL_DIR / f"{name}.joblib")

    metrics = pd.DataFrame(results).set_index("model")
    metrics.to_csv(PROJECT_DIR / "model_metrics.csv")

    test_data = X_test.copy()
    test_data["y"] = y_test.values
    test_data.to_csv(PROJECT_DIR / "test_data.csv", index=False)
    pd.DataFrame([{"feature_count": len(X.columns), "positive_class": "yes"}]).to_csv(
        PROJECT_DIR / "training_metadata.csv", index=False
    )

    print(metrics.round(4))
    print(f"\nSaved 5 models in {MODEL_DIR}")
    print(f"Saved test data with {len(test_data)} rows")


if __name__ == "__main__":
    main()
