import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Step 2: Generating Exploratory Data Analysis...")

# Load the cleaned data
df = pd.read_csv('cleaned_medical_data.csv')
sns.set_theme(style="whitegrid")

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

# 1. The Myth-Buster: Age vs Gender
# First, create age brackets
bins = [0, 18, 35, 60, 100]
labels = ['Children (0-18)', 'Young Adults (19-35)', 'Adults (36-60)', 'Seniors (60+)']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

plt.figure(figsize=(9, 6))
# A pointplot is perfect for showing the exact rate between two groups
sns.pointplot(x='age_group', y='noshow', hue='gender', data=df, palette='Dark2', errorbar=None)
plt.title('No-Show Rates: Age vs. Gender')
plt.xlabel('Age Group')
plt.ylabel('No-Show Rate (%)')
# Map the legend back to text for readability
plt.legend(title='Gender', labels=['Female', 'Male'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# 2. Condition Stacking: How Multiple Illnesses Affect Attendance
# Add up the chronic conditions for each patient
df['total_conditions'] = df['hypertension'] + df['diabetes'] + df['alcoholism'] + df['handicap']

plt.figure(figsize=(8, 5))
sns.barplot(x='total_conditions', y='noshow', data=df, palette='magma', errorbar=None)
plt.title('Impact of Multiple Health Conditions on No-Shows')
plt.xlabel('Total Chronic Conditions (0 to 4)')
plt.ylabel('No-Show Rate (%)')
plt.show()



