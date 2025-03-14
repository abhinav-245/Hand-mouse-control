import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import joblib  # For saving the model

# Load dataset
data = pd.read_csv('gesture_data.csv')
X = data.drop('gesture', axis=1).values  # Features: landmark coordinates
y = data['gesture'].values  # Target: gesture labels

# Normalize the features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate the model
accuracy = knn.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the model
joblib.dump(knn, 'gesture_model.joblib')
print("Model trained and saved as 'gesture_model.joblib'.")
