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
import assemblyai as aai
from requests.exceptions import RequestException
import shutil

# Import EnsembleModel
from models import EnsembleModel

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask + Socket.IO initialization
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'secret!')
socketio = SocketIO(app, cors_allowed_origins="*")

# Paths to saved models
MODEL_PATH = os.path.join('saved_models', 'ensemble_model.pkl')
PIPELINE_PATH = os.path.join('saved_models', 'pipeline.pkl')

# Verify model and pipeline existence
if not os.path.exists(MODEL_PATH) or not os.path.exists(PIPELINE_PATH):
    raise FileNotFoundError("Model or pipeline not found. Please train and save them first.")

# Load model and pipeline
ensemble_model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)

# AssemblyAI configuration
aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')
if not aai.settings.api_key:
    raise ValueError("ASSEMBLYAI_API_KEY is not set in environment variables")
# Instantiate transcriber using updated configuration
transcriber = aai.Transcriber(config=aai.TranscriptionConfig(language_code="en"))

# Paths to executables
YT_DLP_PATH = shutil.which("yt-dlp") or os.path.join(os.getcwd(), "yt-dlp.exe")
FFMPEG_PATH = shutil.which("ffmpeg") or os.path.join(os.getcwd(), "ffmpeg.exe")

# Verify executables exist
if not os.path.exists(YT_DLP_PATH):
    raise FileNotFoundError(f"yt-dlp not found at {YT_DLP_PATH}. Please install it or update the path.")
if not os.path.exists(FFMPEG_PATH):
    raise FileNotFoundError(f"ffmpeg not found at {FFMPEG_PATH}. Please install it or update the path.")

# Cache manager for temporary in-memory storage
class CacheManager:
    def __init__(self):
        self.caches = {}
        self.initialize_caches()

    def initialize_caches(self):
        for cache_type in ['audio', 'news', 'analysis', 'liveStreams']:
            self.caches[cache_type] = {}

    def get(self, type, key):
        cache = self.caches.get(type, {})
        item = cache.get(key)
        if not item or (datetime.now() - item['timestamp']).total_seconds() > item['ttl']:
            return None
        return item['data']

    def set(self, type, key, data, ttl=300):
        if type not in self.caches:
            self.caches[type] = {}
        self.caches[type][key] = {
            'data': data,
            'timestamp': datetime.now(),
            'ttl': ttl
        }

    def delete(self, type, key):
        if type in self.caches and key in self.caches[type]:
            del self.caches[type][key]

    def clear(self):
        self.caches = {}
        self.initialize_caches()

cache_manager = CacheManager()

# Helper function: analyze text
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
        logger.error(f'Error in analyze_text: {e}')
        return {'prediction': 'Error', 'confidence': 0.0}

# Helper function: scrape article
def scrape_article(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'header', 'footer', '.ads', '#comments']):
            element.decompose()
        article = soup.find('article') or soup.find('main') or soup.find('body')
        paragraphs = [p.get_text(strip=True) for p in article.find_all('p') if len(p.get_text(strip=True)) > 30]
        return ' '.join(paragraphs)
    except Exception as e:
        logger.error(f'Error scraping article: {e}')
        return None

# News sources configuration (RSS and GNews)
NEWS_SOURCES = {
    'RSS_FEEDS': [
        {'url': 'https://timesofindia.indiatimes.com/rssfeedstopstories.cms', 'name': 'Times of India', 'reliability': 0.8},
        {'url': 'https://www.thehindu.com/news/national/feeder/default.rss', 'name': 'The Hindu', 'reliability': 0.85}
    ],
    'GNEWS': {
        'endpoint': 'https://gnews.io/api/v4/top-headlines',
        'params': {
            'country': 'in',
            'lang': 'en',
            'max': 10,
            'token': os.getenv('GNEWS_API_KEY')
        }
    }
}

def fetch_gnews_articles():
    api_key = os.getenv('GNEWS_API_KEY')
    if not api_key:
        logger.warning('GNEWS_API_KEY is not set')
        return []
    try:
        response = requests.get(NEWS_SOURCES['GNEWS']['endpoint'], params=NEWS_SOURCES['GNEWS']['params'], timeout=10)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        return articles
    except RequestException as e:
        logger.error(f'GNews API error: {e}')
        return []

def fetch_trending_news():
    cached_news = cache_manager.get('news', 'trending')
    if cached_news:
        return cached_news
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
            is_duplicate = any(difflib.SequenceMatcher(None, item['title'], current['title']).ratio() > 0.8 for item in unique_news)
            if not is_duplicate:
                text = f"{current['title']} {current.get('description', '')}"
                analysis = analyze_text(text)
                unique_news.append({**current, 'analysis': analysis})
        result = unique_news[:15]
        cache_manager.set('news', 'trending', result)
        return result
    except Exception as e:
        logger.error(f'Error fetching trending news: {e}')
        return []

# -----------------------------------------------------------------------------
# Helper: Upload audio buffer to AssemblyAI
# -----------------------------------------------------------------------------
def upload_audio_buffer(audio_buffer):
    """Uploads an in-memory audio buffer to AssemblyAI and returns the upload URL."""
    upload_endpoint = "https://api.assemblyai.com/v2/upload"
    headers = {"authorization": aai.settings.api_key}
    # Ensure the buffer pointer is at the start
    audio_buffer.seek(0)
    response = requests.post(upload_endpoint, headers=headers, data=audio_buffer)
    response.raise_for_status()
    upload_url = response.json()['upload_url']
    return upload_url

# -----------------------------------------------------------------------------
# Recorded Transcription Endpoint (modified)
# -----------------------------------------------------------------------------
@app.route('/api/transcribe-recorded', methods=['POST'])
def transcribe_recorded_route():
    data = request.get_json()
    video_url = data.get('video_url')
    if not video_url:
        return jsonify({"error": "Video URL is required"}), 400

    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to extract audio from {video_url} (Attempt {attempt + 1}/{max_retries})")
            process = subprocess.Popen(
                [YT_DLP_PATH, '-x', '--audio-format', 'mp3', '--output', '-', '--no-playlist', video_url],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            audio_data, error = process.communicate(timeout=120)
            if process.returncode != 0:
                error_msg = error.decode()
                logger.error(f'yt-dlp failed: {error_msg}')
                raise Exception(f'yt-dlp failed: {error_msg}')
            logger.info(f"Audio extracted: {len(audio_data)} bytes")

            # Wrap audio bytes in an in-memory buffer
            audio_buffer = io.BytesIO(audio_data)
            cache_key = f"recorded_{int(time.time() * 1000)}"
            cache_manager.set('audio', cache_key, audio_buffer, ttl=60)

            cached_audio = cache_manager.get('audio', cache_key)
            if not cached_audio:
                raise Exception("Failed to retrieve audio from cache")
            logger.info("Uploading audio to AssemblyAI using in-memory buffer")
            # Use our helper to upload the audio and obtain an upload URL
            upload_url = upload_audio_buffer(cached_audio)
            logger.info("Transcribing audio")
            transcript = transcriber.transcribe(upload_url)
            if transcript.error:
                logger.error(f"Transcription error: {transcript.error}")
                raise Exception(f"Transcription error: {transcript.error}")
            if not transcript.text:
                raise Exception("Transcription returned empty text")
            logger.info("Transcription successful")

            # Remove audio from cache after transcription
            cache_manager.delete('audio', cache_key)

            analysis = analyze_text(transcript.text)
            return jsonify({'text': transcript.text, 'analysis': analysis, 'success': True})

        except subprocess.TimeoutExpired:
            process.terminate()
            logger.error(f"Transcription timed out on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return jsonify({"error": "Transcription timed out after retries"}), 500

        except Exception as e:
            logger.error(f'Transcription error on attempt {attempt + 1}/{max_retries}: {e}')
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

# -----------------------------------------------------------------------------
# Live Stream Processing (modified)
# -----------------------------------------------------------------------------
def start_live_stream(sid, video_url, stop_event):
    """Process live stream by downloading and transcribing 30-second audio segments."""
    max_retries = 3
    segment_duration = 30  # seconds
    while not stop_event.is_set():
        for attempt in range(max_retries):
            try:
                # Get the current stream URL using yt-dlp
                process = subprocess.Popen(
                    [YT_DLP_PATH, '--get-url', '--no-playlist', video_url],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stream_url, error = process.communicate(timeout=30)
                if process.returncode != 0:
                    error_msg = error.decode()
                    logger.error(f'yt-dlp failed: {error_msg}')
                    raise Exception(f'yt-dlp failed: {error_msg}')
                stream_url = stream_url.decode().strip()

                # Record a 30-second audio segment using ffmpeg
                process = subprocess.Popen(
                    [FFMPEG_PATH, '-i', stream_url, '-t', str(segment_duration), '-f', 'mp3', '-'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                audio_data, error = process.communicate(timeout=segment_duration + 30)
                if process.returncode != 0:
                    error_msg = error.decode()
                    logger.error(f'ffmpeg failed: {error_msg}')
                    raise Exception(f'ffmpeg failed: {error_msg}')

                # Wrap the audio in an in-memory buffer and store temporarily
                audio_buffer = io.BytesIO(audio_data)
                cache_key = f"live_{int(time.time() * 1000)}"
                cache_manager.set('audio', cache_key, audio_buffer, ttl=60)

                cached_audio = cache_manager.get('audio', cache_key)
                if not cached_audio:
                    raise Exception("Failed to retrieve live audio from cache")
                logger.info("Uploading live audio to AssemblyAI using in-memory buffer")
                upload_url = upload_audio_buffer(cached_audio)
                transcript = transcriber.transcribe(upload_url)
                if transcript.error:
                    logger.error(f"Transcription error: {transcript.error}")
                    raise Exception(f"Transcription error: {transcript.error}")
                if not transcript.text:
                    logger.warning("Transcription returned empty text")
                    continue

                # Delete cached audio after transcription
                cache_manager.delete('audio', cache_key)

                analysis = analyze_text(transcript.text)
                socketio.emit('transcription', {
                    'text': transcript.text,
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat(),
                    'type': 'Segment'
                }, room=sid)
                break  # Exit retry loop on success

            except Exception as e:
                logger.error(f"Error in live stream processing (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    socketio.emit('error', {'error': str(e)}, room=sid)

        # Wait for the next segment
        time.sleep(segment_duration)

# -----------------------------------------------------------------------------
# Remaining Endpoints and Socket.IO Events
# -----------------------------------------------------------------------------
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
        return jsonify({"error": "Failed to extract article content"}), 500
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

@socketio.on('start_live')
def handle_start_live(data):
    video_url = data.get('url')
    sid = request.sid
    if not video_url:
        socketio.emit('error', {'error': 'Video URL is required'}, room=sid)
        return

    # Stop any existing live stream for this session
    if sid in active_streams:
        active_streams[sid]['stop_event'].set()
        active_streams[sid]['thread'].join()
        del active_streams[sid]

    stop_event = threading.Event()
    thread = threading.Thread(target=start_live_stream, args=(sid, video_url, stop_event))
    thread.start()
    active_streams[sid] = {'thread': thread, 'stop_event': stop_event}
    socketio.emit('status', {'message': f"Started live stream for {video_url}"}, room=sid)

@socketio.on('stop_live')
def handle_stop_live():
    sid = request.sid
    if sid in active_streams:
        active_streams[sid]['stop_event'].set()
        active_streams[sid]['thread'].join()
        del active_streams[sid]
        socketio.emit('status', {'message': 'Live stream stopped'}, room=sid)
    else:
        socketio.emit('error', {'error': 'No active stream to stop'}, room=sid)

if __name__ == '__main__':
    PORT = int(os.getenv('PORT', 3000))
    socketio.run(app, host='0.0.0.0', port=PORT, debug=True)
