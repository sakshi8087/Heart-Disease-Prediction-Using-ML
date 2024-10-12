import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv('heart.csv')

# Preprocess the data: Save a label encoder for each categorical column
encoders = {}
for col in data.select_dtypes('object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le  # Save the encoder for each categorical column

# Save the encoders
for col, encoder in encoders.items():
    joblib.dump(encoder, f'{col}_encoder.pkl')

# Define features and target
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=20)

# Train a Random Forest Classifier
rf = RandomForestClassifier()
params = {
    'criterion': ['gini', 'entropy'],
    'min_samples_split': list(np.arange(2, 101)),
    'min_samples_leaf': list(np.arange(1, 51)),
    'max_depth': list(np.arange(1, 51)),
    'n_estimators': [50, 100, 500, 1000]
}

nrf = RandomizedSearchCV(rf, param_distributions=params, cv=10, n_jobs=-1, scoring='accuracy')
nrf.fit(X_train, y_train)

print(nrf.best_score_)
print(nrf.best_params_)

# Save the trained model
joblib.dump(nrf.best_estimator_, 'heart_disease_model.pkl')
print("Model trained and saved as heart_disease_model.pkl")
