<div align="center">

# Loan Eligibility and EMI Prediction AI

**Loan Eligibility and Safe EMI Prediction using Machine Learning**

An end-to-end supervised learning project that predicts a loan applicant's EMI eligibility and their maximum safe monthly EMI — combining engineered financial ratios with tuned classification and regression models, deployed as an interactive Streamlit app.

**[Live App →](https://loan-eligibility-and-emi-prediction-ai.streamlit.app)**

</div>

---

## Overview

This project analyzes 404,800 loan-applicant records to:

1. **Classify EMI eligibility** into `Eligible`, `High_Risk`, or `Not_Eligible` based on income, expenses, and credit history.
2. **Estimate the maximum safe monthly EMI** an applicant can sustain, so lending decisions can be affordability-based rather than a flat approve/reject.

Both modules, plus supporting analytics, are served through a 4-page Streamlit app (Eligibility Prediction, EMI Calculator, Model Performance, Data Insights).

---

## Data Pipeline

The raw dataset (404,800 rows, 27 columns) was cleaned and engineered as follows:

| Step | Action |
|---|---|
| 1 | Cleaned corrupted numeric strings (age, salary, bank balance) via regex and re-cast to numeric |
| 2 | Standardized inconsistent gender labels to `Male` / `Female` |
| 3 | Set out-of-range credit scores (outside 300–850) to missing, then imputed |
| 4 | Imputed missing `education` (mode), `monthly_rent` (median grouped by `house_type`), and `credit_score` / `bank_balance` / `emergency_fund` (median) |
| 5 | Dropped `family_size` — near-perfectly collinear with `dependents` |
| 6 | Engineered financial ratios: debt-to-income, expense-to-income, disposable income, affordability ratio, EMI-to-income ratio, emergency-fund months, liquidity ratio, and credit-score band |
| 7 | Encoded categoricals — label-mapped binary fields, ordinal-encoded education/credit-score band, one-hot encoded nominal fields |
| 8 | Scaled numerical features with `StandardScaler` |

**Result: 42 engineered features, split 70% / 15% / 15% into train / validation / test (random_state=42).**

---

## Eligibility Classification & Safe EMI Regression

### Dual-Target Supervised Learning

This is a dual-target project: `emi_eligibility` (classification: `Eligible`, `High_Risk`, `Not_Eligible`) and `max_monthly_emi` (regression, in ₹). Both use the same 42 engineered features.

### Model Comparison

Four algorithms were trained and tuned per task — `GridSearchCV` for Logistic Regression, Decision Tree, and Random Forest; `RandomizedSearchCV` for XGBoost — then evaluated on the same held-out test set:

**Classifiers**

| Model | Accuracy | F1 (Macro) ↑ |
|---|---|---|
| **XGBoost** | **97.0%** | **0.897** |
| Decision Tree | 93.3% | 0.800 |
| Random Forest | 92.3% | 0.793 |
| Logistic Regression | 81.9% | 0.669 |

**Regressors**

| Model | RMSE ↓ | R² ↑ |
|---|---|---|
| **XGBoost** | **₹625** | **0.994** |
| Random Forest | ₹1,086 | 0.980 |
| Decision Tree | ₹1,155 | 0.978 |
| Linear Regression | ₹4,100 | 0.720 |

**XGBoost was selected as the final model for both tasks** — it produced the highest accuracy and F1 on classification and the lowest RMSE with the highest R² on regression, outperforming every other candidate on every metric evaluated.

### Test Set Performance (Final XGBoost Models)

| Task | Metric | Score |
|---|---|---|
| Classification | Accuracy | 97.0% |
| Classification | Precision (Macro) | 0.865 |
| Classification | Recall (Macro) | 0.949 |
| Classification | F1-score (Macro) | 0.897 |
| Classification | ROC-AUC (OVR) | 0.997 |
| Regression | RMSE | ₹625 |
| Regression | MAE | ₹229 |
| Regression | R² Score | 0.9935 |

Eligibility classes are imbalanced (`Not_Eligible` majority, `High_Risk` minority), handled via class-weighted / sample-weighted training across all classifiers. Experiment runs across all model/task combinations were tracked with MLflow.

---

## Repository Structure

```
EMI-Predict-AI/
├── Home.py                                    # Streamlit application entry point
├── notebooks/
│   ├── EMIPredict_AI_Model_Development.ipynb  # Full analysis: EDA, feature engineering, modeling, evaluation
│   └── images/                                # Saved chart exports from the notebook
├── pages/
│   ├── 1_Eligibility_Prediction.py
│   ├── 2_EMI_Calculator.py
│   ├── 3_Model_Performance.py
│   └── 4_Data_Insights.py
├── utils/
│   ├── model_loader.py
│   ├── form_inputs.py
│   └── preprocessing.py
├── models/
│   ├── final_classifier.joblib                # Trained XGBoost classifier
│   ├── final_regressor.joblib                 # Trained XGBoost regressor
│   ├── scaler.joblib                          # StandardScaler fit on numerical features
│   ├── label_encoder.joblib                   # Encoders for categorical features
│   └── feature_names.joblib                   # Final engineered feature list
├── assets/                                    # App images/icons
├── requirements.txt                           # Python dependencies
└── README.md
```

---

## Running Locally

```bash
git clone <repository-url>
cd Loan-Eligibility-and-EMI-Prediction-AI
pip install -r requirements.txt
streamlit run Home.py
```

The app expects the trained artifacts (`final_classifier.joblib`, `final_regressor.joblib`, `scaler.joblib`, `label_encoder.joblib`, `feature_names.joblib`) inside `models/`. These are produced by running `notebooks/EMIPredict_AI_Model_Development.ipynb` end-to-end, or can be used as already provided in this repo.

---

## Tech Stack

- **Data / ML:** pandas, numpy, scikit-learn (Logistic Regression, Decision Tree, Random Forest), XGBoost
- **Visualization:** matplotlib, seaborn
- **Experiment Tracking:** MLflow
- **App:** Streamlit
- **Model persistence:** joblib

---

# Author

**MOHIT SINGH RAJPUT — AI/ML Engineer**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/mohitsingh1307)
[![GitHub](https://img.shields.io/badge/GitHub-121011?style=flat-square&logo=github&logoColor=white)](https://github.com/Mohit-1307)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/mohitsinghrajput1307)
[![LeetCode](https://img.shields.io/badge/LeetCode-181717?style=flat-square&logo=leetcode&logoColor=FFA116)](https://leetcode.com/u/MOHIT_SINGH_RAJPUT/)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:mohitsinghrajput1307@gmail.com)

---

<div align="center">

*If this project was useful, a ⭐ on the repository is appreciated.*

</div>