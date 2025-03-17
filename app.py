import pandas as pd
import re
import nltk
import json
import os
import random
import time
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources
nltk.download('vader_lexicon')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Function to clean tweets
def clean_tweet(tweet):
    return ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(r\w+:\/\/\S+)", " ", tweet).split())

# Custom sentiment analyzer with crypto-specific enhancements
class CryptoSentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.sia.lexicon.update({
            'bullish': 3.0, 'hodl': 2.0, 'mooning': 3.0, 'to the moon': 3.0,
            'adoption': 2.0, 'staking': 1.5, 'yield': 1.5, 'airdrop': 2.0,
            'bearish': -3.0, 'dump': -2.5, 'rugpull': -4.0, 'scam': -3.5,
            'hack': -3.0, 'exploit': -3.0, 'crash': -3.0, 'fud': -2.0
        })
    
    def analyze(self, text):
        return self.sia.polarity_scores(text)

# Function for sentiment analysis
def get_sentiment(tweet, analyzer):
    analysis = analyzer.analyze(tweet)
    if analysis['compound'] > 0.05:
        return 'Positive'
    elif analysis['compound'] < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Function to generate mock crypto tweets
def get_mock_crypto_tweets(count=50):
    tweets_data = []
    now = datetime.now()
    tokens = ['BTC', 'ETH', 'SOL', 'AVAX', 'ADA', 'DOT']
    
    for i in range(count):
        tweet_content = f"{random.choice(tokens)} is trending! Should I buy more?"
        tweet_date = (now - timedelta(days=random.randint(0, 7))).strftime('%Y-%m-%d %H:%M:%S')
        tweets_data.append({
            'date': tweet_date,
            'content': tweet_content,
            'username': f"user_{random.randint(1000, 9999)}"
        })
    
    return tweets_data

# Function to analyze tweets
def analyze_tweets(df):
    if df is None or df.empty:
        return None

    analyzer = CryptoSentimentAnalyzer()
    df['cleaned_content'] = df['content'].apply(clean_tweet)
    df['sentiment_score'] = df['cleaned_content'].apply(lambda x: analyzer.analyze(x)['compound'])
    df['sentiment'] = df['cleaned_content'].apply(lambda x: get_sentiment(x, analyzer))
    return df

# API to fetch sentiment analysis
@app.route('/api/sentiment', methods=['GET'])
def get_sentiment_data():
    try:
        tweets_data = get_mock_crypto_tweets(count=50)
        df = pd.DataFrame(tweets_data)
        analyzed_df = analyze_tweets(df)
        if analyzed_df is not None:
            return jsonify(analyzed_df.to_dict(orient='records'))
        return jsonify({"error": "No tweets found"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API to export data as CSV
@app.route('/api/export-csv', methods=['GET'])
def export_csv():
    try:
        tweets_data = get_mock_crypto_tweets(count=50)
        df = pd.DataFrame(tweets_data)
        analyzed_df = analyze_tweets(df)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"crypto_sentiment_{timestamp}.csv"
        analyzed_df.to_csv(filename, index=False)
        return send_file(filename, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check API
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "version": "1.0"}), 200

# Run Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
