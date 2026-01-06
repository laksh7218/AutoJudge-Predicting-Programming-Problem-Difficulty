import os
# 1. FIX CPU ERROR: Force joblib to use 1 core
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

import joblib
import warnings
import re
import numpy as np
import streamlit as st
from scipy.sparse import hstack
import pandas as pd
from sklearn.base import BaseEstimator

# 2. CONFIG MUST BE FIRST
st.set_page_config(
    page_title="Problem Difficulty Predictor",
    layout="centered"
)

# Ignore warnings
warnings.filterwarnings("ignore")

# Define Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 3. FIX MEMORY CRASH: Use Caching
# This prevents the app from reloading models every time you click a button
@st.cache_resource
def load_models():
    try:
        tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        reg_model = joblib.load(os.path.join(MODEL_DIR, "reg_model.pkl"))
        clf_model = joblib.load(os.path.join(MODEL_DIR, "clf_model.pkl"))
        label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        return tfidf, scaler, reg_model, clf_model, label_encoder
    except Exception as e:
        return None, None, None, None, None

# Load the models once
tfidf, scaler, reg_model, clf_model, label_encoder = load_models()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# Add fake junk samples
junk_samples = [
    "asdfgh qwerty",
    "random text only",
    "hello how are you",
    "this is not a coding problem",
    "abcdefg xyz"
]

junk_df = pd.DataFrame({
    "clean_text": junk_samples,
    "is_valid": 0
})
# Stop if models failed
if tfidf is None:
    st.error("🚨 Error: Could not load models. Please ensure the 'models' folder exists on GitHub.")
    st.stop()

# Reject meaningless input
if len(clean.split()) < 15:
    st.error("❌ Input text is too short or not meaningful.")
    st.stop()

if X_tfidf.nnz == 0:
    st.error("❌ Input does not contain recognizable problem content.")
    st.stop()

# ... (The rest of your code: functions and UI logic remains exactly the same) ...

st.title(" Problem Difficulty Predictor")
st.caption("Paste a competitive programming problem to predict its difficulty.")


# FEATURE FUNCTIONS 

def count_math_symbols(text):
    if not isinstance(text, str):
        return 0
    symbols = [
        "<=", ">=", "==", "!=", "+", "-", "*", "/", "%",
        "<", ">", "=", "^", "|", "&",
        "(", ")", "{", "}", "[", "]"
    ]
    return sum(text.count(sym) for sym in symbols)

def constraint_count(text):
    patterns = [
        r"constraints",
        r"1\s*<=\s*n",
        r"10\^",
        r"<=\s*10",
        r"seconds?",
        r"memory"
    ]
    return sum(len(re.findall(p, text.lower())) for p in patterns)


def io_complexity(text):
    score = 0
    score += text.lower().count("input")
    score += text.lower().count("output")
    score += text.lower().count("test case")
    score += text.lower().count("multiple")
    return score


def example_count(text):
    return len(re.findall(r"example", text.lower()))


keyword_weights = {
    "dp":5, "dynamic programming":5,
    "segment tree":5, "fenwick":5,
    "bitmask":4,
    "graph":3, "dfs":3, "bfs":3,
    "recursion":3,
    "binary search":2,
    "greedy":2,
    "math":1
}

def weighted_keyword_score(text):
    return sum(text.count(k) * w for k,w in keyword_weights.items())



keywords = [
    "graph", "tree", "dp", "dynamic programming", "recursion",
    "backtracking", "greedy", "binary search",
    "heap", "priority queue", "segment tree", "fenwick",
    "bitmask", "math", "modulo"
]

def keyword_frequency(text, keywords):
    return sum(text.count(kw) for kw in keywords)


def keyword_diversity(text, keywords):
    found = {kw for kw in keywords if kw in text}
    return len(found)

# UI
st.subheader(" Enter Problem Details")
problem_desc = st.text_area("Problem Description", height=150)
input_desc = st.text_area("Input Description", height=120)
output_desc = st.text_area("Output Description", height=120)


# PREDICTION
if st.button(" Predict", use_container_width=True):

    #  Empty input check
    if not problem_desc and not input_desc and not output_desc:
        st.warning("Please enter at least one field.")
        st.stop()

    #  Combine input
    full_text = problem_desc + " " + input_desc + " " + output_desc

    # Clean text
    clean = clean_text(full_text)

    #  Meaningful length check
    if len(clean.split()) < 15:
        st.error("❌ Input text is too short or not meaningful.")
        st.stop()

    #  TF-IDF transform
    X_tfidf = tfidf.transform([clean])

    #  Vocabulary check
    if X_tfidf.nnz == 0:
        st.error("❌ Input does not contain recognizable problem content.")
        st.stop()

    #  Numeric feature extraction
    s_count = len(re.findall(r'[.!?]', full_text))
    t_len = len(full_text)
    avg_s_len = t_len / (s_count + 1)
    m_sym = count_math_symbols(full_text)
    m_den = m_sym / (t_len + 1)
    c_cnt = constraint_count(full_text)
    io_cx = io_complexity(full_text)
    ex_cnt = example_count(full_text)
    k_weight = weighted_keyword_score(clean)
    k_div = keyword_diversity(clean, keywords)

    s_comp = c_cnt + io_cx + (m_den * 10) + k_weight

    user_features = [[
        s_count, avg_s_len, m_den, c_cnt, io_cx, ex_cnt,
        k_weight, k_div, s_comp
    ]]

    numeric_scaled = scaler.transform(user_features)

    X_user = hstack([X_tfidf, numeric_scaled])

#  Predictions
    score_pred = reg_model.predict(X_user.toarray())[0]
    class_pred = clf_model.predict(X_user)[0]
    class_label = label_encoder.inverse_transform([class_pred])[0]

    #  Output
    st.markdown("---")
    st.subheader(" Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Difficulty Score", f"{score_pred:.2f}")

    with col2:
        emoji = {"easy": "🟢", "medium": "🟠", "hard": "🔴"}.get(
            class_label.lower(), "⚪"
        )
        st.metric("Predicted Difficulty Class", f"{emoji} {class_label.upper()}")


