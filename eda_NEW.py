import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Load Data
df = pd.read_excel("sensing_intuitive.xlsx")

# standardize column names
df.columns = df.columns.str.strip().str.lower()
print(f"Columns found: {df.columns.tolist()}")

# define label column
df['label'] = df['sensing'].apply(lambda x: 'Sensing' if x == 1 else 'Intuitive')


# define post column
text_col = 'post'

# stats
total = len(df)
counts = df['label'].value_counts()
percent = df['label'].value_counts(normalize=True) * 100

print("\n--- DATASET STATS ---")
print(f"Total Examples: {total}")
print(f"Intuitive Count: {counts.get('Intuitive', 0)} ({percent.get('Intuitive', 0):.2f}%)")
print(f"Sensing Count: {counts.get('Sensing', 0)} ({percent.get('Sensing', 0):.2f}%)")

# word counts
df['word_count'] = df[text_col].astype(str).apply(lambda x: len(x.split()))
avg_words = df['word_count'].mean()
print(f"Avg Words per Example: {avg_words:.0f}")

# vocabulary size
print("Calculating vocabulary size...")
vec = CountVectorizer(min_df=5, stop_words='english')
vec.fit(df[text_col].fillna("").astype(str))
print(f"Vocabulary Size (Unique tokens appearing >5 times): {len(vec.get_feature_names_out())}")