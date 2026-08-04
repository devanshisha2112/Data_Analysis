# Task 2 : Exploratory Data Analysis (EDA)
# Dataset : Titanic Dataset

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set graph style
sns.set(style="whitegrid")

print("=" * 60)
print("      EXPLORATORY DATA ANALYSIS - TITANIC DATASET")
print("=" * 60)

# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Display First and Last Records
print("\nFirst 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())

# Dataset Information
print("\nDataset Shape")
print(df.shape)

print("\nNumber of Rows :", df.shape[0])
print("Number of Columns :", df.shape[1])

print("\nColumn Names")
print(df.columns)

print("\nData Types")
print(df.dtypes)

print("\nDataset Information")
print(df.info())

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Duplicate Records
print("\nDuplicate Rows")
print(df.duplicated().sum())

# Statistical Summary
print("\nStatistical Summary")
print(df.describe())

# Unique Values
print("\nUnique Values in Sex")
print(df["Sex"].unique())

print("\nUnique Values in Passenger Class")
print(df["Pclass"].unique())

# Ask Meaningful Questions
print("\n================ QUESTIONS ================")

print("\n1. How many passengers survived?")
print(df["Survived"].value_counts())

print("\n2. How many Male and Female passengers?")
print(df["Sex"].value_counts())

print("\n3. How many passengers were in each class?")
print(df["Pclass"].value_counts())

print("\n4. Average Age of Passengers")
print(df["Age"].mean())

print("\n5. Average Fare")
print(df["Fare"].mean())

print("\n6. Maximum Fare")
print(df["Fare"].max())

print("\n7. Minimum Fare")
print(df["Fare"].min())

# Group Analysis
print("\nAverage Age by Gender")
print(df.groupby("Sex")["Age"].mean())

print("\nAverage Fare by Passenger Class")
print(df.groupby("Pclass")["Fare"].mean())

print("\nSurvival Rate by Gender")
print(df.groupby("Sex")["Survived"].mean())

print("\nSurvival Rate by Passenger Class")
print(df.groupby("Pclass")["Survived"].mean())

# Correlation
print("\nCorrelation Matrix")
print(df.corr(numeric_only=True))

# VISUALIZATIONS

# Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# Gender Count
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", data=df)
plt.title("Gender Count")
plt.show()

# Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", data=df)
plt.title("Passenger Class")
plt.show()

# Age Distribution
plt.figure(figsize=(8,5))
plt.hist(df["Age"].dropna(), bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Fare Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Fare"], kde=True)
plt.title("Fare Distribution")
plt.show()

# Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()

# Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.show()

# Box Plot
plt.figure(figsize=(7,5))
sns.boxplot(x="Pclass", y="Fare", data=df)
plt.title("Fare by Passenger Class")
plt.show()

# Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(x="Age", y="Fare", hue="Survived", data=df)
plt.title("Age vs Fare")
plt.show()

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# Conclusion

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("""
Key Findings:

1. Female passengers had a higher survival rate than males.

2. First-class passengers survived more than second and third class passengers.

3. Some columns contain missing values such as Age and Cabin.

4. Ticket Fare has some very high values (outliers).

5. Most passengers were between 20 and 40 years old.

6. Most passengers travelled in Third Class.

7. The dataset is now well understood and ready for further analysis.
""")