import pandas as pd
import pickle

# load cleaned data
df = pd.read_csv("data/cleaned/combined_cleaned.csv")

# load saved model
with open("outputs/sentiment_model_embeddings.pkl", "rb") as f:
    clf = pickle.load(f)

# load embedder
with open("outputs/embedder.pkl", "rb") as f:
    embedder = pickle.load(f)

# generate embeddings
X = embedder.encode(df["title_clean"].tolist())

# predict sentiment
df["predicted_sentiment"] = clf.predict(X)

# save results
df.to_csv("outputs/sentiment_results.csv", index=False)

print("Predictions saved to outputs/sentiment_results.csv")

import pandas as pd

df = pd.read_csv("outputs/sentiment_results.csv")
print(df["predicted_sentiment"].value_counts())
