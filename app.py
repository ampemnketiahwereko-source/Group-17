import pandas as pd
import streamlit as st

st.title("Customer Support NLP Pipeline")

# Load image files
st.header("EDA Dashboard")
st.image("eda_dashboard.png")

st.header("Baseline Confusion Matrix")
st.image("baseline_confusion_matrix.png")

# Load data files
st.header("Model Comparison Results")
df_comparison = pd.read_csv("model_comparison_results.csv")
st.dataframe(df_comparison)

st.header("Preprocessed Prompts Data")
df_prompts = pd.read_csv("bitext_preprocessed_prompts.csv")
st.dataframe(df_prompts.head())
