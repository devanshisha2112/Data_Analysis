# Task 3 : Data Visualization
# Dataset : Titanic Dataset

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Graph Style
sns.set(style="whitegrid")

print("============================================================")
print("          TITANIC DATA VISUALIZATION")
print("=========================================================" )

# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

print("\nDataset Loaded Successfully!")
print("Rows :", df.shape[0])
print("Columns :", df.shape[1])

# 1. Survival Count

plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Passenger Survival Count")
plt.xlabel("Survival (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

# 2. Gender Distribution

plt.figure(figsize=(6,4))
sns.countplot(x="Sex", data=df)
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# 3. Passenger Class Distribution

plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", data=df)
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()

# 4. Age Distribution

plt.figure(figsize=(8,5))
plt.hist(df["Age"].dropna(), bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# 5. Fare Distribution

plt.figure(figsize=(8,5))
sns.histplot(df["Fare"], kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

# 6. Survival by Gender

plt.figure(figsize=(7,5))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Passengers")
plt.legend(["Did Not Survive", "Survived"])
plt.show()

# 7. Survival by Passenger Class

plt.figure(figsize=(7,5))
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Passengers")
plt.legend(["Did Not Survive", "Survived"])
plt.show()

# 8. Box Plot (Fare by Passenger Class)

plt.figure(figsize=(8,5))
sns.boxplot(x="Pclass", y="Fare", data=df)
plt.title("Fare by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.show()

# 9. Scatter Plot (Age vs Fare)

plt.figure(figsize=(8,5))
sns.scatterplot(x="Age", y="Fare", hue="Survived", data=df)
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

# 10. Correlation Heatmap

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm",
            linewidths=0.5)

plt.title("Correlation Heatmap")
plt.show()

# 11. Pie Chart

survival = df["Survived"].value_counts()

plt.figure(figsize=(6,6))
plt.pie(
    survival,
    labels=["Not Survived", "Survived"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Passenger Survival Percentage")
plt.show()

# Final Conclusion

print("================================================================" )
print("DATA VISUALIZATION COMPLETED SUCCESSFULLY")
print("================================================================" )

print("""

Insights:

1. Most passengers did not survive.

2. Male passengers were more than female passengers.

3. Most passengers travelled in Third Class.

4. Female passengers had a higher survival rate.

5. First Class passengers survived more frequently.

6. Ticket Fare contains some very high-value outliers.

7. Most passengers were between 20 and 40 years old.

8. Visualizations clearly reveal trends, comparisons, and relationships.

""")