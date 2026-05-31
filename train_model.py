import pandas as pd
import joblib


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

print("Step 3: Training the Machine Learning Model...")

# Load the cleaned data
df = pd.read_csv('cleaned_medical_data.csv')

X = df.drop('noshow', axis=1)
y = df['noshow']

# Split data (using random_state=42 is critical so the evaluation script uses the same split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize model
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)

# Hyperparameter tuning setup
param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, None],
    'min_samples_split': [5, 10]
}

random_search = RandomizedSearchCV(
    estimator=rf_model, param_distributions=param_dist, n_iter=5, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42
)

# Train the model
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Save the trained model to a file
joblib.dump(best_model, 'hospital_noshow_model.pkl')
print(f"Model training complete. Saved to 'hospital_noshow_model.pkl'.\nBest Params: {random_search.best_params_}")