import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import joblib
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# LOAD DATA
df = pd.read_csv(
    r"C:\Users\sapna\Downloads\problems_data.csv")

print(df.isnull().sum())
print(df.duplicated().sum())
duplicate_indices = df.index[df.duplicated()].tolist()
print(f"Duplicates are at row numbers: {duplicate_indices}")

df["final_description"] = (
    df["title"].fillna("") +
    df["sample_io"].fillna("").astype(str) +
    df["url"].fillna("").astype(str) +
    df["description"].fillna("") +
    df["input_description"].fillna("") +
    df["output_description"].fillna("")
)

#  dropping 
df.drop(
    columns=[
        "description", "input_description", "output_description",
        "title", "sample_io", "url"
    ],
    inplace=True
)
print(df.info())
df['text_length'] = df['final_description'].apply(len)

def count_math_symbols(text):
    if not isinstance(text, str):
        return 0
    symbols = [
        "<=", ">=", "==", "!=", "+", "-", "*", "/", "%",
        "<", ">", "=", "^", "|", "&",
        "(", ")", "{", "}", "[", "]"
    ]
    return sum(text.count(sym) for sym in symbols)
df['math_symbol_count'] = df['final_description'].apply(count_math_symbols)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

df['clean_text'] = df['final_description'].apply(clean_text)
df['sentence_count'] = df['final_description'].apply(
    lambda x: len(re.findall(r'[.!?]', str(x)))
)
df['avg_sentence_length'] = (
    df['text_length'] / (df['sentence_count'] + 1)
)
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

df['constraint_count'] = df['final_description'].apply(constraint_count)
def io_complexity(text):
    score = 0
    score += text.lower().count("input")
    score += text.lower().count("output")
    score += text.lower().count("test case")
    score += text.lower().count("multiple")
    return score

df['io_complexity'] = df['final_description'].apply(io_complexity)
def example_count(text):
    return len(re.findall(r"example", text.lower()))

df['example_count'] = df['final_description'].apply(example_count)
df['math_density'] = (
    df['math_symbol_count'] / (df['text_length'] + 1)
)
# KEYWORD FEATURES
keywords = [
    "graph", "tree", "dp", "dynamic programming", "recursion",
    "backtracking", "greedy", "binary search",
    "heap", "priority queue", "segment tree", "fenwick",
    "bitmask", "math", "modulo"
]

def keyword_frequency(text, keywords):
    return sum(text.count(kw) for kw in keywords)

df['keyword_count'] = df['clean_text'].apply(
    lambda x: keyword_frequency(x, keywords)
)
def keyword_diversity(text, keywords):
    found = {kw for kw in keywords if kw in text}
    return len(found)

df['keyword_diversity'] = df['clean_text'].apply(
    lambda x: keyword_diversity(x, keywords)
)
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

df["keyword_weighted_score"] = df["clean_text"].apply(weighted_keyword_score)
# TF-IDF FEATURES
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=3000,
    ngram_range=(1, 2)
)

X_tfidf = tfidf.fit_transform(df['clean_text'])
print("TF-IDF shape:", X_tfidf.shape)
word_counts = pd.Series(
    " ".join(df['clean_text']).split()
).value_counts()

df['structural_complexity'] = (
    df['constraint_count'] +
    df['io_complexity'] +
    df['math_density'] * 10 +
    df['keyword_weighted_score']
)


# COMBINING FEATURES
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(
    df[['sentence_count',
    'avg_sentence_length',
    'math_density',
    'constraint_count',
    'io_complexity',
    'example_count',
    'keyword_weighted_score',
    'keyword_diversity',
    'structural_complexity']]
)

X_final = hstack([X_tfidf, numeric_scaled])
print("Final feature matrix shape:", X_final.shape)
le = LabelEncoder()
y_clf = le.fit_transform(df['problem_class'])
X_train, X_test, y_clf_train, y_clf_test = train_test_split(
    X_final,
    y_clf,
    test_size=0.2,
    random_state=42
)


from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# SVM
svm_clf = LinearSVC(
    C=0.5,
    class_weight="balanced",
    max_iter=5000
)
svm_clf.fit(X_train, y_clf_train)
y_clf_pred = svm_clf.predict(X_test)

# Logistic Regression
log_reg = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)
log_reg.fit(X_train, y_clf_train)
y_log_pred = log_reg.predict(X_test)

# Random Forest
rf_clf = RandomForestClassifier(
    n_estimators=800,
    max_depth=50,
    class_weight="balanced",
    random_state=42,
)
rf_clf.fit(X_train, y_clf_train)
y_rf_pred = rf_clf.predict(X_test)


# CLASSIFICATION RESULTS
print("\nCLASSIFICATION RESULTS")
print("SVM Accuracy:", accuracy_score(y_clf_test, y_clf_pred))
print("LogReg Accuracy:", accuracy_score(y_clf_test, y_log_pred))
print("RF Accuracy:", accuracy_score(y_clf_test, y_rf_pred))

print("\nConfusion Matrix (SVM):", confusion_matrix(y_clf_test, y_clf_pred))
print("\nConfusion Matrix (LogReg):", confusion_matrix(y_clf_test, y_log_pred))
print("\nConfusion Matrix (RF):", confusion_matrix(y_clf_test, y_rf_pred))