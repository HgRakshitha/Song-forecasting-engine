import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
import joblib

# Load the dataset
df = pd.read_csv('ex.csv')

# Preprocess the data
df['User-Rating'] = df['User-Rating'].str.extract(r'(\d+\.\d+)').astype(float)
df['User-ID'] = df.index
df = df[['User-ID', 'Song-Name', 'User-Rating']]

# Transform the data into a suitable format
reader = Reader(rating_scale=(7.0, 10.0))
dataset = Dataset.load_from_df(df[['User-ID', 'Song-Name', 'User-Rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.2)

# Hyperparameter tuning for SVD
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

# Save the model
joblib.dump(best_model, 'best_svd_model.pkl')

# Save the dataset
df.to_pickle('musicrec.pkl')

# Evaluate the best SVD model
svd_predictions = best_model.test(testset)
svd_rmse = accuracy.rmse(svd_predictions, verbose=True)
svd_mae = accuracy.mae(svd_predictions, verbose=True)

print(f'\nSVD Model Evaluation:\nRMSE: {svd_rmse}\nMAE: {svd_mae}')
