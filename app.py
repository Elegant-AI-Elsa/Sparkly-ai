from flask import Flask, request, jsonify, render_template
from scraper import scrape_website
from vector_store import create_collection, store_text
from gemini_qa import get_answer
import asyncio
import threading

# Patch: Set default event loop for non-main thread (Flask request handlers)
if not asyncio.get_event_loop_policy().get_event_loop():
    asyncio.set_event_loop(asyncio.new_event_loop())

def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

threading.current_thread().run = lambda: ensure_event_loop()


app = Flask(__name__)

# Homepage
@app.route('/')
def home():
    return render_template('index.html')  # Make sure templates/index.html exists

# Train the model on given website URL
@app.route('/train', methods=['POST'])
def train():
    url = request.json.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        text = scrape_website(url)
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        create_collection()
        store_text(chunks)
        return jsonify({'message': 'Training completed successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ask a question
@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        answer = get_answer(question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
