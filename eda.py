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
<<<<<<< Updated upstream
=======

# 1. Simple Histogram of Patient Ages
plt.figure(figsize=(8, 5))
sns.histplot(df['age'], bins=20, kde=False, color='skyblue')
plt.title('Distribution of Patient Ages')
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.show()

# 2. Gender vs. No-Show
plt.figure(figsize=(7, 5))
sns.countplot(x='gender', hue='noshow', data=df, palette='Set2')
plt.title('Appointment Status by Gender')
plt.xlabel('Gender (0 = Female, 1 = Male)')
plt.ylabel('Total Count')
plt.legend(title='Status', labels=['Showed Up', 'No-Show'])
plt.show()

# 3. Scholarship (Financial Aid) vs. No-Show Rate
plt.figure(figsize=(7, 5))
sns.barplot(x='scholarship', y='noshow', data=df, palette='pastel', errorbar=None)
plt.title('Does Financial Aid Affect No-Shows?')
plt.xlabel('Receives Scholarship (0 = No, 1 = Yes)')
plt.ylabel('No-Show Rate (%)')
plt.show()


>>>>>>> Stashed changes
