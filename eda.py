import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Step 2: Generating Exploratory Data Analysis...")

# Load the cleaned data
df = pd.read_csv('cleaned_medical_data.csv')
sns.set_theme(style="whitegrid")

# Plot 1: Lead Time Impact
plt.figure(figsize=(10, 6))
sns.boxplot(x='noshow', y='lead_time', data=df, palette='Set2')
plt.title('How Waiting Days (Lead Time) Affects No-Shows')
plt.xlabel('No-Show (0 = No, 1 = Yes)')
plt.ylabel('Lead Time (Days)')
plt.show()

# Plot 2: Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation")
plt.show()
