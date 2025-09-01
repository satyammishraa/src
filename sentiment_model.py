from transformers import pipeline

def load_sentiment_model(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    return pipeline("sentiment-analysis", model=model_name)

def analyze_sentiment(model, texts):
    results = model(texts, truncation=True)
    return results
