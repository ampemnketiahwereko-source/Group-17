import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

st.title("Customer Support NLP Pipeline")

@st.cache_resource
def train():
    df = pd.read_csv("bitext_customer_support_dataset.csv")
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(df["instruction"])
    clf = LogisticRegression(max_iter=1000).fit(X, df["intent"])
    return df, vec, X, clf

def get_reply(message):
    df, vec, X, clf = train()
    q = vec.transform([message])
    intent = clf.predict(q)[0]
    rows = np.where(df["intent"].values == intent)[0]
    sims = cosine_similarity(q, X[rows]).ravel()
    best = rows[sims.argmax()]
    return intent, df["response"].iloc[best]

tab_chat, tab_results = st.tabs(["Chatbot", "Project Results"])

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
    if prompt := st.chat_input("Ask a customer support question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        intent, answer = get_reply(prompt)
        with st.chat_message("assistant"):
            st.caption(f"Detected intent: {intent}")
            st.write(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": f"*Detected intent: {intent}*\n\n{answer}"}
        )

with tab_results:
    st.header("EDA Dashboard")
    st.image("eda_dashboard.png")
    st.header("Baseline Confusion Matrix")
    st.image("baseline_confusion_matrix.png")
    st.header("Model Comparison Results")
    st.dataframe(pd.read_csv("model_comparison_results.csv"))
    st.header("Preprocessed Prompts Data")
    st.dataframe(pd.read_csv("bitext_preprocessed_prompts.csv", nrows=5))
