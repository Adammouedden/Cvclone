from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
except Exception:
    CORS = None
import os
from dotenv import load_dotenv
load_dotenv()

# Import the helper from chat_cli
from chat_cli import generate_reply

app = Flask(__name__)
# Enable CORS if flask_cors is available. This allows the frontend (served on another port)
# to send POST requests and the browser preflight (OPTIONS) to succeed.
if CORS:
    CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/chat/civilian', methods=['POST'])
def civilian_chat():
    data = request.get_json() or {}
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    try:
        reply = generate_reply(text, civilian=1)
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/api/chat/enterprise', methods=['POST'])
def enterprise_chat():
    data = request.get_json() or {}
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    try:
        reply = generate_reply(text, civilian=0)
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('CHAT_SERVER_PORT', 5001))
    app.run(host='0.0.0.0', port=port)
