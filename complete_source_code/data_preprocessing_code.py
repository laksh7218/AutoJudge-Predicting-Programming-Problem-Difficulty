import pandas as pd
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