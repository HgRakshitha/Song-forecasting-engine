import pandas as pd
import numpy as np
from surprise import Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load the dataset
df = pd.read_pickle('musicrec.pkl')

# Transform the data into a suitable format for Surprise
reader = Reader(rating_scale=(7.0, 10.0))
dataset = Dataset.load_from_df(df[['User-ID', 'Song-Name', 'User-Rating']], reader)

# Create a pivot table
pivot_table = df.pivot(index='User-ID', columns='Song-Name', values='User-Rating').fillna(0)

# Compute the cosine similarity matrix
item_similarity = cosine_similarity(pivot_table.T)
item_similarity_df = pd.DataFrame(item_similarity, index=pivot_table.columns, columns=pivot_table.columns)

# Save the similarity matrix
joblib.dump(item_similarity_df, 'similarities.pkl')
