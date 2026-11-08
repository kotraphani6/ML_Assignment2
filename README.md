# Bank Marketing Classification - ML Assignment 2

This Streamlit application trains and evaluates five classifiers on the UCI Bank Marketing dataset:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Support Vector Machine

## Dataset

The dataset contains bank campaign records. The target is `y`, where `yes` means the customer subscribed to a term deposit and `no` means they did not. The positive class for binary metrics is `yes`.

## Project structure

- `train_models.py`: loads and prepares the raw data, trains all models, and saves artifacts
- `app.py`: Streamlit evaluation interface
- `test_data.csv`: reproducible held-out test set
- `model_metrics.csv`: comparison of saved-model performance
- `model/`: serialized trained pipelines
- `requirements.txt`: Python dependencies

## Setup and run

```bash
cd "ML-ASSINNMENT2-08-11-2026"
python -m pip install -r requirements.txt
python train_models.py
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. The app can evaluate the bundled test data or an uploaded CSV containing the bank customer feature columns and a `y` column.

## Reproduce results

Training uses an 80/20 stratified split with random seed 42. Numeric features are standardized for Logistic Regression, KNN, and SVM. Models are saved with `joblib` and reported metrics include Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient.
