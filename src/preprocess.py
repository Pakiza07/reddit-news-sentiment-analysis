import pandas as pd
from utils import clean_text
import os

os.makedirs("data/cleaned", exist_ok=True)

# Read Reddit CSV
df_reddit = pd.read_csv("data/raw/reddit_raw.csv")
df_reddit['title_clean'] = df_reddit['title'].apply(clean_text)
df_reddit['text_clean'] = df_reddit['text'].apply(clean_text)

# Read News CSV
df_news = pd.read_csv("data/raw/news_articles_20260304_201356.csv")
df_news['title_clean'] = df_news['title'].apply(clean_text)
df_news['text_clean'] = df_news['text'].apply(clean_text)

# Combine datasets
df_combined = pd.concat([
    df_reddit[['title_clean', 'text_clean', 'url']].rename(columns={'text_clean': 'text'}),
    df_news[['title_clean', 'text_clean', 'url']].rename(columns={'title_clean':'title','text_clean':'text'})
], ignore_index=True)

# Save cleaned CSV
df_combined.to_csv("data/cleaned/combined_cleaned.csv", index=False)
print("Cleaned and combined dataset ready for NLP!")
