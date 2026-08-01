# Task 4 : Sentiment Analysis
# Dataset : Customer Reviews

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Graph Style
sns.set(style="whitegrid")

print("=====================================================")
print("          SENTIMENT ANALYSIS PROJECT")
print("=====================================================")

# Load Dataset
df = pd.read_csv("reviews.csv")

print("\nDataset Loaded Successfully!\n")
print(df.head())

# Function to Detect Sentiment
def get_sentiment(text):

    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return "Positive"

    elif polarity < 0:
        return "Negative"

    else:
        return "Neutral"

# Apply Sentiment Analysis
df["Sentiment"] = df["Review"].apply(get_sentiment)

# Display Result
print("\nReviews with Sentiment\n")
print(df)

# Count Sentiments
print("\nSentiment Count\n")
print(df["Sentiment"].value_counts())

# Percentage
percentage = round(
    df["Sentiment"].value_counts(normalize=True) * 100,
    2
)

print("\nSentiment Percentage\n")
print(percentage)

# BAR CHART

plt.figure(figsize=(6,4))

sns.countplot(x="Sentiment", data=df)

plt.title("Sentiment Count")

plt.xlabel("Sentiment")

plt.ylabel("Number of Reviews")

plt.show()

# PIE CHART

counts = df["Sentiment"].value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    counts,
    labels=counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Sentiment Distribution")

plt.show()

# Save Result

df.to_csv("sentiment_output.csv", index=False)

print("\nResult saved as sentiment_output.csv")

# Conclusion

print("=====================================================")

print("SENTIMENT ANALYSIS COMPLETED SUCCESSFULLY")

print("=====================================================")

print("""

Insights:

1. Positive reviews indicate customer satisfaction.

2. Negative reviews highlight customer complaints.

3. Neutral reviews express balanced opinions.

4. Businesses can use these insights to improve products.

5. Sentiment analysis helps companies understand customer feedback quickly.

""")