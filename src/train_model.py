import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
from sentence_transformers import SentenceTransformer
import os

# ------------------------
# Load combined CSV
# ------------------------
data = pd.read_csv("data/cleaned/combined_cleaned.csv") 

# ------------------------
# Combine text columns into one
# ------------------------
data["combined_text"] = data["title_clean"].fillna("") + " " + data["text"].fillna("")

# ------------------------
# Generate sentiment labels (VADER)
# ------------------------
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

data["label"] = data["combined_text"].apply(get_sentiment)

# Optional: see class distribution
print("Label counts:\n", data["label"].value_counts())

# ------------------------
# Define features and target
# ------------------------
X = data["combined_text"]
y = data["label"]

# ------------------------
# Train/test split
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# Generate sentence embeddings
# ------------------------
os.makedirs("outputs", exist_ok=True)
model_name = "all-MiniLM-L6-v2"  # lightweight, fast model
embedder = SentenceTransformer(model_name)

X_train_vec = embedder.encode(X_train.tolist(), show_progress_bar=True)
X_test_vec = embedder.encode(X_test.tolist(), show_progress_bar=True)

# ------------------------
# Train Logistic Regression
# ------------------------
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# ------------------------
# Evaluate model
# ------------------------
y_pred = clf.predict(X_test_vec)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# ------------------------
# Save model and embedder
# ------------------------
pickle.dump(clf, open("outputs/sentiment_model_embeddings.pkl", "wb"))
pickle.dump(embedder, open("outputs/embedder.pkl", "wb"))

# ------------------------
# Test a manual example
# ------------------------
sample = ["The new AI policy will greatly improve economic growth."]
sample_vec = embedder.encode(sample)
prediction = clf.predict(sample_vec)
print("\nSample Prediction:", prediction)
