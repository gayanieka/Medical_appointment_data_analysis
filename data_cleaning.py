import pandas as pd
import numpy as np

print("Step 1: Starting Data Cleaning...")

# Load raw data
df = pd.read_csv('KaggleV2-May-2016.csv')

# Standardize column names
df.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap', 'No-show': 'NoShow'}, inplace=True)
df.columns = df.columns.str.lower()

# Parse datetimes and engineer 'lead_time'
df['scheduledday'] = pd.to_datetime(df['scheduledday'])
df['appointmentday'] = pd.to_datetime(df['appointmentday'])
df['lead_time'] = (df['appointmentday'].dt.date - df['scheduledday'].dt.date).dt.days

# Remove anomalies
df = df[df['lead_time'] >= 0] 
df = df[(df['age'] >= 0) & (df['age'] <= 100)] 
df['handicap'] = df['handicap'].clip(upper=1) 

# Encode categorical variables
df['noshow'] = df['noshow'].map({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].map({'M': 1, 'F': 0})

# Drop unused columns
df.drop(['patientid', 'appointmentid', 'neighbourhood', 'scheduledday', 'appointmentday'], axis=1, inplace=True)

# Save the cleaned dataset for the next scripts
df.to_csv('cleaned_medical_data.csv', index=False)
print("Data cleaning complete. Saved to 'cleaned_medical_data.csv'.")