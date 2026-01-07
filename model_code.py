import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import joblib
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

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

X = df[['final_description']]
y1 = df[['problem_score']]
y2 = df[['problem_class']]

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


# SCATTERPLOTS
plt.scatter(df['text_length'], df['problem_score'])
plt.xlabel('text_length')
plt.ylabel("Problem Score")
plt.show()

plt.scatter(df['text_length'], df['problem_class'])
plt.xlabel('text_length')
plt.ylabel("Problem class")
plt.show()

color_map = {'easy': 'green', 'medium': 'orange', 'hard': 'red'}
colors = df['problem_class'].map(color_map)

plt.figure(figsize=(8, 5))
plt.scatter(df['text_length'], df['problem_score'], c=colors)
plt.xlabel("Text Length")
plt.ylabel("Problem Score")
plt.title("Text Length vs Problem Score (colored by Problem Class)")
for cls, color in color_map.items():
    plt.scatter([], [], c=color, label=cls)
plt.legend()
plt.show()


# SYMBOL FEATURES
df['math_symbol_count'] = df['final_description'].apply(count_math_symbols)

plt.scatter(df['math_symbol_count'], df['problem_score'])
plt.xlabel("Number of Mathematical Symbols")
plt.ylabel("Problem Score")
plt.show()

plt.scatter(df['math_symbol_count'], df['problem_class'])
plt.xlabel("Number of Mathematical Symbols")
plt.ylabel("Problem class")
plt.show()

fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot problem_score (Regression target)
ax1.scatter(
    df['math_symbol_count'],
    df['problem_score'],
    color='tab:blue',
    label='Problem Score'
)
ax1.set_xlabel("Number of Mathematical Symbols")
ax1.set_ylabel("Problem Score", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Second Y-axis for problem_class (Classification target)
ax2 = ax1.twinx()
ax2.scatter(
    df['math_symbol_count'],
    df['problem_class'],
    color='tab:red',
    label='Problem Class'
)
ax2.set_ylabel("Problem Class", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title("Relationship Between Math Symbol Count and Targets")
plt.show()
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

plt.scatter(df['keyword_count'], df['problem_score'])
plt.xlabel("Keyword Frequency")
plt.ylabel("Problem Score")
plt.show()

plt.scatter(df['keyword_count'], df['problem_class'])
plt.xlabel("Keyword Frequency")
plt.ylabel("Problem class")
plt.show()

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


# LABEL ENCODING
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_reg = df['problem_score'].values
y_clf = le.fit_transform(df['problem_class'])


# TRAIN-TEST SPLIT
from sklearn.model_selection import train_test_split

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X_final,
    y_reg,
    y_clf,
    test_size=0.2,
    random_state=42
)

# REGRESSION MODELS
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# HistGradientBoosting
hgbr = HistGradientBoostingRegressor(
    max_iter=800,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
hgbr.fit(X_train.toarray(), y_reg_train)
y_reg_pred = hgbr.predict(X_test.toarray())

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_reg_train)
y_lin_pred = lin_reg.predict(X_test)

# Random Forest
rf_reg = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_train, y_reg_train)
y_rf_reg_pred = rf_reg.predict(X_test)


# REGRESSION RESULTS
print("\nREGRESSION RESULTS")
print("HistGB R2:", r2_score(y_reg_test, y_reg_pred))
print("Linear R2:", r2_score(y_reg_test, y_lin_pred))
print("RF R2:", r2_score(y_reg_test, y_rf_reg_pred))

print("HistGB RMSE:", np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)))
print("Linear RMSE:",np.sqrt(mean_squared_error(y_reg_test, y_lin_pred)))
print("RF RMSE:", np.sqrt(mean_squared_error(y_reg_test, y_rf_reg_pred)))


# CLASSIFICATION MODELS
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
print("\nConfusion Matrix (SVM):", confusion_matrix(y_clf_test, y_log_pred))
print("\nConfusion Matrix (SVM):", confusion_matrix(y_clf_test, y_rf_pred))



# COMPARISON TABLES
regression_results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "HistGradientBoosting"],
    "R2 Score": [
        r2_score(y_reg_test, y_lin_pred),
        r2_score(y_reg_test, y_rf_reg_pred),
        r2_score(y_reg_test, y_reg_pred)
    ]
})

classification_results = pd.DataFrame({
    "Model": ["Logistic Regression", "Linear SVM", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_clf_test, y_log_pred),
        accuracy_score(y_clf_test, y_clf_pred),
        accuracy_score(y_clf_test, y_rf_pred)
    ]
})

print("\nRegression Comparison")
print(regression_results.sort_values("R2 Score", ascending=False))

print("\nClassification Comparison")
print(classification_results.sort_values("Accuracy", ascending=False))


os.makedirs("models", exist_ok=True)

joblib.dump(tfidf, "models/tfidf.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(hgbr, "models/reg_model.pkl")   # regression model
joblib.dump(svm_clf, "models/clf_model.pkl")  # classification model
joblib.dump(le, "models/label_encoder.pkl")

print(" PKL files created successfully!")

