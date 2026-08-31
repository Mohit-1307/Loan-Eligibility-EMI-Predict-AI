import streamlit as st

pg = st.navigation(
    [
        st.Page("pages/home.py", title="Home", icon=":material/home:"),
        st.Page(
            "pages/eligibility_prediction.py",
            title="Eligibility Prediction",
            icon=":material/verified:",
        ),
        st.Page(
            "pages/emi_calculator.py",
            title="EMI Calculator",
            icon=":material/calculate:",
        ),
        st.Page(
            "pages/model_performance.py",
            title="Model Performance",
            icon=":material/insights:",
        ),
        st.Page(
            "pages/data_insights.py", title="Data Insights", icon=":material/bar_chart:"
        ),
    ]
)

pg.run()
