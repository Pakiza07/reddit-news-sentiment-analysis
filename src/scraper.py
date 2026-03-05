import requests
import pandas as pd
from newspaper import Article
from bs4 import BeautifulSoup
import logging
import os
from datetime import datetime
import time

# ---------------------
# Logging Setup
# ---------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='logs/scraper.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------------
# Reddit Loading Function
# ---------------------
def scrape_subreddit(subreddit_name, limit=200):
    headers = {
        "User-Agent": "reddit-sentiment-analysis-app"
    }

    url = f"https://www.reddit.com/r/{subreddit_name}/hot.json?limit={limit}"

    response = requests.get(url, headers=headers)
    data = response.json()

    posts = []

    for post in data["data"]["children"]:
        post_data = post["data"]

        posts.append({
            "id": post_data.get("id"),
            "title": post_data.get("title"),
            "text": post_data.get("selftext"),
            "created_utc": post_data.get("created_utc"),
            "score": post_data.get("score"),
            "num_comments": post_data.get("num_comments"),
            "url": "https://www.reddit.com" + post_data.get("permalink", "")
                })

    df = pd.DataFrame(posts)
    return df

# ---------------------
# Reddit Scraper for multiple subreddits
# ---------------------
def scrape_subreddits(subreddit_list, limit_per_subreddit=200, pause=2):
    all_posts = []
    for subreddit in subreddit_list:
        print(f"Scraping /r/{subreddit} ...")
        df = scrape_subreddit(subreddit, limit=limit_per_subreddit)
        df["subreddit"] = subreddit
        all_posts.append(df)
        time.sleep(pause)
    combined_df = pd.concat(all_posts, ignore_index=True)
    return combined_df

# ---------------------
# News Article Scraper
# ---------------------
def scrape_news_article(url, retries=3, delay=2):
    for attempt in range(retries):
        try:
            article = Article(url)
            article.download()
            article.parse()
            return {"title": article.title, "text": article.text, "url": url}
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(delay)
    logging.error(f"All attempts failed for {url}")
    return None

# ---------------------
# Save Data Function
# ---------------------
def save_dataframe(df, name_prefix):
    os.makedirs("data", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join("data", f"{name_prefix}_{timestamp}.csv")
    df.to_csv(file_path, index=False)
    logging.info(f"Saved {file_path}")
    return file_path

# ---------------------
# Example Usage
# ---------------------
if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    # List of subreddits to scrape
    subreddits_to_scrape = ["wallstreetbets", "stocks", "investing", "technology"]

    df_reddit = scrape_subreddits(subreddits_to_scrape, limit_per_subreddit=200)

    df_reddit.to_csv("data/raw/reddit_raw.csv", index=False)

    print("Reddit scraping done!")
    print("Rows:", len(df_reddit))

    # ----- News Example -----
    news_urls = [
        "https://www.bbc.com/news/articles/crrxx5x7wyko",
        "https://www.bbc.com/news/articles/cwy884ekn0jo",
        "https://www.nytimes.com/2026/03/04/world/asia/china-ai-enthusiasm.html"
    ]
    news_data = []
    for url in news_urls:
        article = scrape_news_article(url)
        if article:
            news_data.append(article)

    if news_data:
        df_news = pd.DataFrame(news_data)
        save_dataframe(df_news, "news_articles")
        print("News scraping done!")