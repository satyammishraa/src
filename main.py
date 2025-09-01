from src.scraper import scrape_reviews
from src.sentiment_model import load_sentiment_model, analyze_sentiment
import pandas as pd

def main():
    # Step 1: Scrape reviews (or load CSV)
    # df = scrape_reviews("https://example.com/reviews", pages=3)
    df = pd.read_csv("data/reviews_sample.csv")  # fallback sample

    # Step 2: Load sentiment model
    model = load_sentiment_model()

    # Step 3: Run sentiment analysis
    results = analyze_sentiment(model, df["review"].tolist()[:10])
    df["sentiment"] = [r["label"] for r in results]
    df["score"] = [r["score"] for r in results]

    print(df.head())
    df.to_csv("data/reviews_with_sentiment.csv", index=False)

if __name__ == "__main__":
    main()
