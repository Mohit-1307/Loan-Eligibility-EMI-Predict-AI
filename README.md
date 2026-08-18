# EMIPredict AI - Intelligent Financial Risk Assessment Platform

A dual machine learning system for EMI (Equated Monthly Installment) underwriting:

1. **Classification** - predicts whether an applicant is `Eligible`, `High_Risk`, or `Not_Eligible` for the EMI they're requesting.
2. **Regression** - predicts `max_monthly_emi`, the maximum EMI the applicant can safely afford.

Both models are trained on a 404,800-row applicant dataset and served through a multi-page Streamlit application.

## Project Structure

```
EMIPredict_AI/
├── Home.py                          # Streamlit app entry point
├── pages/
│   ├── 1_Eligibility_Prediction.py  # Classifier - eligibility + probabilities
│   ├── 2_EMI_Calculator.py          # Regressor - safe EMI estimate
│   ├── 3_Model_Performance.py       # Model comparison, confusion matrix, feature importance
│   └── 4_Data_Insights.py           # Historical trends + live segment explorer
├── utils/
│   ├── preprocessing.py             # Feature engineering - mirrors the notebook exactly
│   ├── form_inputs.py               # Shared applicant input form
│   └── model_loader.py              # Cached model/data loading
├── models/                          # Trained models + preprocessing objects (.joblib)
├── notebooks/
│   └── EMIPredict_AI_Model_Development.ipynb   # Full EDA, feature engineering, and modeling
├── data/
│   └── emi_prediction_dataset.csv
├── assets/                          # Chart images used by the Model Performance / Data Insights pages
├── requirements.txt
└── .streamlit/config.toml
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the Notebook

```bash
jupyter notebook notebooks/EMIPredict_AI_Model_Development.ipynb
```

Covers data cleaning, EDA, hypothesis testing, feature engineering, model training (4 classifiers, 4 regressors), MLflow tracking, and saving the final models consumed by the app.

## Running the App

```bash
streamlit run Home.py
```

## Experiment Tracking (MLflow)

```bash
mlflow ui
```

Then open `http://localhost:5000` to compare every run logged from the notebook's MLflow section.

## Deploying to Streamlit Cloud

1. Push this repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at `Home.py`.
3. Streamlit Cloud installs `requirements.txt` automatically - no other configuration needed.

Note: `data/emi_prediction_dataset.csv` is ~75 MB. If your GitHub plan or Streamlit Cloud instance has trouble with a file that size, use [Git LFS](https://git-lfs.com) for that one file, or host it externally and download it in a setup step - the app itself only needs `data/emi_prediction_dataset.csv` for the Data Insights page; `models/*.joblib` is what the two prediction pages actually run on.

## Model Results (test set)

| Task | Model | Key metric |
|---|---|---|
| Classification | Decision Tree (deployed) | 93.3% accuracy, 0.80 macro-F1 |
| Classification | Random Forest | 91.8% accuracy, 0.98 ROC-AUC |
| Classification | Logistic Regression (baseline) | 81.9% accuracy |
| Regression | Decision Tree (deployed) | RMSE ₹1,084, R² 0.98 |
| Regression | Random Forest | RMSE ₹1,086, R² 0.98 |
| Regression | Linear Regression (baseline) | RMSE ₹4,100, R² 0.72 |

Both deployed models clear the project targets (>90% classification accuracy, <₹2,000 regression RMSE). Full methodology, charts, and the reasoning behind every modeling decision are in the notebook.

## About the XGBoost and MLflow Sections

This project was built in a sandboxed environment with no internet access, so the `xgboost` and `mlflow` packages could not be installed there. Their code in the notebook is written and ready to run (standard, well-established APIs), but was **not executed** in that environment - those specific cells carry no output, and are clearly marked as such in the notebook. Six of the eight models (every model except the two XGBoost ones), all 23 charts, both hypothesis test sections, and the entire preprocessing pipeline **were** executed for real, with genuine results.

Once you run `pip install -r requirements.txt` in your own environment, every cell - including XGBoost and MLflow - runs as part of a normal top-to-bottom execution. If XGBoost turns out to outperform the Decision Tree models already in `models/`, retrain and overwrite `models/final_classifier.joblib` / `models/final_regressor.joblib` with it - the app loads whatever model is at those two file paths, so no application code needs to change.

## Dataset

`data/emi_prediction_dataset.csv` - 404,800 rows, 27 columns, covering demographics, employment, housing, expenses, credit history, and the requested loan. The raw file has realistic data-quality issues (corrupted numeric strings, inconsistent category labels, out-of-range credit scores, missing values) that are cleaned in the notebook - see Section 3 (Data Wrangling) for the full detail on what was found and how it was fixed.
