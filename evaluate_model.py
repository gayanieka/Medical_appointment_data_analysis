import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

print("Step 4: Evaluating the Model...")

# Load the cleaned data and the trained model
df = pd.read_csv('cleaned_medical_data.csv')
model = joblib.load('hospital_noshow_model.pkl')

# Re-create the EXACT test set used in Script 3
X = df.drop('noshow', axis=1)
y = df['noshow']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Make predictions
y_pred = model.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Show', 'No-Show'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Feature Importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title("Feature Importance: Drivers of No-Shows")
plt.show()
