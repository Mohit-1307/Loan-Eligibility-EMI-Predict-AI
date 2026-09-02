import joblib
import streamlit as st
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "emi_prediction.csv"
MODELS_DIR = ROOT_DIR / "models"


@st.cache_resource
def get_models():
    """Load every trained model and preprocessing object once, cached for the session."""

    classifier = joblib.load(MODELS_DIR / "final_classifier.joblib")
    regressor = joblib.load(MODELS_DIR / "final_regressor.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")

    return {
        "classifier": classifier,
        "regressor": regressor,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
    }


@st.cache_data
def get_dataset_sample(n=20000):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    if len(df) > n:
        df = df.sample(n, random_state=42)

    return df
