import io
import os
import re
import time
import json
import difflib
import joblib
import requests
import feedparser
import logging
import subprocess
import threading
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
from requests.exceptions import RequestException
import shutil

# Import EnsembleModel from models.py
from models import EnsembleModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask + Socket.IO setup
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'secret!')
socketio = SocketIO(app, cors_allowed_origins="*")

# Paths
MODEL_PATH = os.path.join('saved_models', 'ensemble_model.pkl')
PIPELINE_PATH = os.path.join('saved_models', 'pipeline.pkl')

# Safe model loader
def safe_joblib_load(path):
    try:
        with open(path, 'rb') as f:
            return joblib.load(f)
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        raise

# Load models safely
ensemble_model = safe_joblib_load(MODEL_PATH)
pipeline = safe_joblib_load(PIPELINE_PATH)

# Verify yt-dlp and ffmpeg
YT_DLP_PATH = shutil.which("yt-dlp") or os.path.join(os.getcwd(), "yt-dlp")
FFMPEG_PATH = shutil.which("ffmpeg") or os.path.join(os.getcwd(), "ffmpeg")

if not os.path.exists(YT_DLP_PATH):
    raise FileNotFoundError(f"yt-dlp not found at {YT_DLP_PATH}")
if not os.path.exists(FFMPEG_PATH):
    raise FileNotFoundError(f"ffmpeg not found at {FFMPEG_PATH}")

# AssemblyAI setup
import assemblyai as aai
aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')
if not aai.settings.api_key:
    raise ValueError("ASSEMBLYAI_API_KEY is not set.")
transcriber = aai.Transcriber(config=aai.TranscriptionConfig(language_code="en"))

# Cache manager
class CacheManager:
    def __init__(self):
        self.caches = {type: {} for type in ['audio', 'news', 'analysis', 'liveStreams']}

    def get(self, type, key):
        cache = self.caches.get(type, {})
        item = cache.get(key)
        if item and (datetime.now() - item['timestamp']).total_seconds() <= item['ttl']:
            return item['data']
        return None

    def set(self, type, key, data, ttl=300):
        self.caches[type][key] = {'data': data, 'timestamp': datetime.now(), 'ttl': ttl}

    def delete(self, type, key):
        if key in self.caches.get(type, {}):
            del self.caches[type][key]

cache_manager = CacheManager()

# Helper: Analyze text
def analyze_text(text):
    if not text.strip():
        return {'prediction': 'False', 'confidence': 0.0}
    try:
        X = pipeline.transform([text])
        prediction = ensemble_model.predict(X)
        confidence = ensemble_model.predict_proba(X)[:, 1][0]
        return {
            'prediction': 'True' if prediction[0] == 1 else 'False',
            'confidence': float(confidence)
        }
    except Exception as e:
        logger.error(f"Error in analyze_text: {e}")
        return {'prediction': 'Error', 'confidence': 0.0}

# Helper: Scrape article
def scrape_article(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        article = soup.find('article') or soup.find('main') or soup.find('body')
        paragraphs = [p.get_text(strip=True) for p in article.find_all('p') if len(p.get_text(strip=True)) > 30]
        return ' '.join(paragraphs)
    except Exception as e:
        logger.error(f"Error scraping article: {e}")
        return None

# Upload audio to AssemblyAI
def upload_audio_buffer(audio_buffer):
    audio_buffer.seek(0)
    response = requests.post(
        "https://api.assemblyai.com/v2/upload",
        headers={"authorization": aai.settings.api_key},
        data=audio_buffer
    )
    response.raise_for_status()
    return response.json()['upload_url']

# RSS + GNews fetching
NEWS_SOURCES = {
    'RSS_FEEDS': [
        {'url': 'https://timesofindia.indiatimes.com/rssfeedstopstories.cms', 'name': 'Times of India', 'reliability': 0.8},
        {'url': 'https://www.thehindu.com/news/national/feeder/default.rss', 'name': 'The Hindu', 'reliability': 0.85}
    ],
    'GNEWS': {
        'endpoint': 'https://gnews.io/api/v4/top-headlines',
        'params': {'country': 'in', 'lang': 'en', 'max': 10, 'token': os.getenv('GNEWS_API_KEY')}
    }
}

def fetch_gnews_articles():
    if not NEWS_SOURCES['GNEWS']['params']['token']:
        return []
    try:
        response = requests.get(NEWS_SOURCES['GNEWS']['endpoint'], params=NEWS_SOURCES['GNEWS']['params'], timeout=10)
        response.raise_for_status()
        return response.json().get('articles', [])
    except RequestException as e:
        logger.error(f"GNews API error: {e}")
        return []

def fetch_trending_news():
    cached = cache_manager.get('news', 'trending')
    if cached:
        return cached
    try:
        rss_results = []
        for source in NEWS_SOURCES['RSS_FEEDS']:
            feed = feedparser.parse(source['url'])
            for item in feed.entries[:10]:
                rss_results.append({
                    'title': item.title,
                    'description': item.get('description', item.get('summary', '')),
                    'url': item.link,
                    'source': source['name'],
                    'reliability': source['reliability'],
                    'published': item.get('published', '')
                })
        gnews_results = fetch_gnews_articles()
        all_news = rss_results + gnews_results

        unique_news = []
        for current in all_news:
            is_duplicate = any(
                difflib.SequenceMatcher(None, item['title'], current['title']).ratio() > 0.8
                for item in unique_news
            )
            if not is_duplicate:
                text = f"{current['title']} {current.get('description', '')}"
                analysis = analyze_text(text)
                unique_news.append({**current, 'analysis': analysis})

        result = unique_news[:15]
        cache_manager.set('news', 'trending', result)
        return result
    except Exception as e:
        logger.error(f"Error fetching trending news: {e}")
        return []

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze-article', methods=['POST'])
def analyze_article():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({"error": "Article URL is required"}), 400
    article_text = scrape_article(url)
    if not article_text:
        return jsonify({"error": "Failed to extract article"}), 500
    analysis = analyze_text(article_text)
    return jsonify({'text': article_text, 'analysis': analysis, 'success': True})

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text_route():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "Text is required"}), 400
    analysis = analyze_text(text)
    return jsonify({'text': text, 'analysis': analysis, 'success': True})

@app.route('/api/trending-news', methods=['GET'])
def trending_news_route():
    return jsonify(fetch_trending_news())

@app.route('/api/news-stream')
def news_stream_route():
    def generate():
        while True:
            news = fetch_trending_news()
            yield f"data: {json.dumps(news)}\n\n"
            time.sleep(10)
    return app.response_class(generate(), mimetype='text/event-stream')

# Socket.IO handlers
active_streams = {}

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in active_streams:
        active_streams[sid]['stop_event'].set()
        active_streams[sid]['thread'].join()
        del active_streams[sid]
    logger.info('Client disconnected')

if __name__ == '__main__':
    PORT = int(os.getenv('PORT', 3000))
    socketio.run(app, host='0.0.0.0', port=PORT)
