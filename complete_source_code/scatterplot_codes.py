import pandas as pd
import matplotlib.pyplot as plt
import os
import re

os.environ['LOKY_MAX_CPU_COUNT'] = '1'

# LOAD DATA
df = pd.read_csv(
    r"C:\Users\sapna\Downloads\dataset.csv")

duplicate_indices = df.index[df.duplicated()].tolist()

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
