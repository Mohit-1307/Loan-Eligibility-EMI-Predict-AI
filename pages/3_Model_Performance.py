"""
EMIPredict AI - Model Performance Page

Shows how the deployed models were evaluated - metric comparison across
every model that was trained, confusion matrix, actual-vs-predicted, and
feature importance. All figures come straight from the model-development
notebook's test-set evaluation.
"""

import streamlit as st
from pathlib import Path

st.set_page_config(

    page_title = "Model Performance - EMIPredict AI",

    page_icon = ":material/insights:",

    layout = "wide"

)

ASSETS_DIR = Path(__file__).resolve().parent.parent / 'assets'

st.title(":material/insights: Model Performance")

st.caption("All metrics below are computed on a held-out test set the models never saw during training or tuning.")

tab_clf, tab_reg = st.tabs([":material/verified: Eligibility Classifier", ":material/calculate: EMI Regressor"])

with tab_clf:

    st.subheader(":material/compare_arrows: Model Comparison")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", "93.3%", icon = ":material/target:")

    c2.metric("Precision (macro)", "0.775", icon = ":material/adjust:")

    c3.metric("Recall (macro)", "0.864", icon = ":material/replay:")

    c4.metric("F1-score (macro)", "0.800", icon = ":material/balance:")

    st.image(str(ASSETS_DIR / 'model_comparison_clf.png'), use_container_width = True)

    st.caption(
        
        "XGBoost is part of the model set but is not shown here - it was not trained in the environment "
        "this app was built in (see README). The deployed classifier is the Decision Tree, the strongest "
        "of the three models that were trained and compared."
    
    )

    c1, c2 = st.columns(2)

    with c1:

        st.subheader(":material/grid_on: Confusion Matrix")

        st.image(str(ASSETS_DIR / 'cm_decision_tree.png'), use_container_width = True)

    with c2:

        st.subheader(":material/ssid_chart: Feature Importance")

        st.image(str(ASSETS_DIR / 'featimp_clf_dt.png'), use_container_width = True)

    st.caption(
        
        "The engineered ratio features (EMI-to-income, disposable income, proposed installment, "
        "affordability ratio) dominate - direct confirmation that the feature engineering in the notebook "
        "is what drives the model's decisions, not any single raw input."
    
    )

with tab_reg:

    st.subheader(":material/compare_arrows: Model Comparison")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("RMSE", "\u20b91,084", icon = ":material/straighten:")

    c2.metric("MAE", "\u20b9371", icon = ":material/rule:")

    c3.metric("R\u00b2 Score", "0.980", icon = ":material/analytics:")

    c4.metric("Target", "< \u20b92,000 RMSE", icon = ":material/flag:")

    st.image(str(ASSETS_DIR / 'model_comparison_reg.png'), use_container_width = True)

    st.caption(
        
        "XGBoost is part of the model set but is not shown here for the same reason as the classifier tab. "
        "The deployed regressor is the Decision Tree, matched almost exactly by Random Forest."
        
    )

    c1, c2 = st.columns(2)

    with c1:

        st.subheader(":material/scatter_plot: Actual vs Predicted")

        st.image(str(ASSETS_DIR / 'avp_decision_tree.png'), use_container_width = True)

    with c2:

        st.subheader(":material/ssid_chart: Feature Importance")

        st.image(str(ASSETS_DIR / 'featimp_reg_dt.png'), use_container_width = True)

st.divider()

st.subheader(":material/menu_book: Full Methodology")

st.write(

    "Data cleaning, feature engineering, hyperparameter tuning and every evaluation shown here are documented "
    "step by step in `notebooks/EMIPredict_AI_Model_Development.ipynb`, including the exact train/validation/test "
    "split and the reasoning behind each modeling decision."

)