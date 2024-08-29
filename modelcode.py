#best accuracy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split, GridSearchCV, cross_validate

# Step 1: Load the dataset
df = pd.read_csv('ex.csv')

# Step 2: Preprocess the data
df['User-Rating'] = df['User-Rating'].str.extract(r'(\d+\.\d+)').astype(float)
df['User-ID'] = df.index
df = df[['User-ID', 'Song-Name', 'User-Rating']]

# Step 3: Transform the data into a suitable format
reader = Reader(rating_scale=(7.0, 10.0))
dataset = Dataset.load_from_df(df[['User-ID', 'Song-Name', 'User-Rating']], reader)

# Step 4: Split the data into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.2)

# Step 5: Hyperparameter tuning for SVD
param_grid = {
    'n_factors': [10, 20, 50, 100],
    'n_epochs': [20, 30, 40, 50],
    'lr_all': [0.002, 0.005, 0.01, 0.02],
    'reg_all': [0.02, 0.05, 0.1]
}

gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5)
gs.fit(dataset)

# Best RMSE score and parameters
print(f'Best RMSE score: {gs.best_score["rmse"]}')
print(f'Best parameters: {gs.best_params["rmse"]}')

# Use the best SVD model
best_model = gs.best_estimator['rmse']
best_model.fit(trainset)

# Evaluate the best SVD model with cross-validation
cross_val_results = cross_validate(best_model, dataset, measures=['rmse'], cv=5, verbose=True)
print(f'Mean RMSE from cross-validation: {cross_val_results["test_rmse"].mean()}')

# Step 6: Train KNNBasic model for finding similar songs
sim_options = {
    'name': 'cosine',
    'user_based': False
}

knn_model = KNNBasic(sim_options=sim_options)
knn_model.fit(trainset)

# Function to recommend top N songs similar to a given song
def recommend_songs(model, song_name, n=5):
    try:
        song_inner_id = model.trainset.to_inner_iid(song_name)
        song_neighbors = model.get_neighbors(song_inner_id, k=n)
        song_neighbors = [model.trainset.to_raw_iid(inner_id) for inner_id in song_neighbors]
        return song_neighbors
    except ValueError:
        print(f"Item '{song_name}' is not part of the trainset.")
        return []

# List some available songs from the training set using knn_model
available_songs = [knn_model.trainset.to_raw_iid(iid) for iid in range(len(knn_model.trainset._raw2inner_id_items))]

# Print some available songs
print("Available songs in the training set (first 10):")
print(available_songs[:10])

# Select a song from the training set for recommendations
selected_song = available_songs[0]  # Replace with any valid song name from the list
print(f"\nSelected song for recommendations: {selected_song}")

# Get recommendations
recommended_songs = recommend_songs(knn_model, selected_song, n=5)
print(f'Songs recommended similar to "{selected_song}": {recommended_songs}')

# Evaluation of models
print("\nEvaluating models...")

# Evaluate SVD model
print("Evaluating SVD model...")
svd_predictions = best_model.test(testset)
svd_rmse = accuracy.rmse(svd_predictions, verbose=True)
svd_mae = accuracy.mae(svd_predictions, verbose=True)

# Evaluate KNN model
print("Evaluating KNN model...")
knn_predictions = knn_model.test(testset)
knn_rmse = accuracy.rmse(knn_predictions, verbose=True)
knn_mae = accuracy.mae(knn_predictions, verbose=True)

# Print comparison
print("\nComparison of Models:")
print(f"SVD - RMSE: {svd_rmse}, MAE: {svd_mae}")
print(f"KNN - RMSE: {knn_rmse}, MAE: {knn_mae}")

# Visualization: Performance Metrics
metrics = {
    'Model': ['SVD', 'KNN'],
    'RMSE': [svd_rmse, knn_rmse],
    'MAE': [svd_mae, knn_mae]
}
metrics_df = pd.DataFrame(metrics)

# RMSE Plot
plt.figure(figsize=(8, 6))
sns.barplot(x='Model', y='RMSE', data=metrics_df, palette='viridis')
plt.title('RMSE of Models')
plt.show()

# MAE Plot
plt.figure(figsize=(8, 6))
sns.barplot(x='Model', y='MAE', data=metrics_df, palette='viridis')
plt.title('MAE of Models')
plt.show()

# Rating Distribution Plot
plt.figure(figsize=(8, 6))
sns.histplot(df['User-Rating'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of User Ratings')
plt.xlabel('User Rating')
plt.ylabel('Frequency')
plt.show()

# Song Frequencies Plot
song_counts = df['Song-Name'].value_counts()
plt.figure(figsize=(10, 8))
sns.barplot(x=song_counts.index, y=song_counts.values, palette='viridis')
plt.title('Frequency of Songs Rated')
plt.xlabel('Song Name')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()
