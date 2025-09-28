from flask import Flask, request, jsonify, make_response, send_from_directory
try:
    from flask_cors import CORS
except Exception:
    CORS = None
import os
from agents.base_agent import Agent

app = Flask(__name__)
if CORS:
    CORS(app, resources={r"/api/*": {"origins": "*"}})

civilian_agent = Agent(civilian=1)
enterprise_agent = Agent(civilian=0)

# --- Image upload config ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # keep consistent with mimetype check
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.after_request
def add_cors_headers(response):
    response.headers.setdefault('Access-Control-Allow-Origin', '*')
    response.headers.setdefault('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.setdefault('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    return response

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, conditional=True)

# ---------- Text chat ----------
@app.route('/api/chat/civilian', methods=['POST', 'OPTIONS'])
def civilian_chat():
    if request.method == 'OPTIONS':
        return make_response('', 200)
    data = request.get_json() or {}
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    try:
        reply = civilian_agent.generate_reply(text)
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/enterprise', methods=['POST', 'OPTIONS'])
def enterprise_chat():
    if request.method == 'OPTIONS':
        return make_response('', 200)
    data = request.get_json() or {}
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    try:
        reply = enterprise_agent.generate_reply(text)
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------- Image chat ----------
def _is_allowed_mime(mime: str) -> bool:
    return mime in ('image/png', 'image/jpeg')

@app.route('/api/chat/civilian_images', methods=['POST', 'OPTIONS'])  # fixed spelling
def upload_civilian_image():
    if request.method == 'OPTIONS':
        return make_response('', 200)

    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image provided (field name "image")'}), 400

    if not _is_allowed_mime((file.mimetype or '').lower()):
        return jsonify({'error': 'Unsupported type (use PNG or JPEG)'}), 415

    image_bytes = file.read()
    try:
        resp = civilian_agent.respond_to_image(image_bytes)
        # ensure we return JSON/dict, not a raw string
        return jsonify({'reply': resp}) if not isinstance(resp, (dict, list)) else jsonify(resp)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/enterprise_images', methods=['POST', 'OPTIONS'])
def upload_enterprise_image():
    if request.method == 'OPTIONS':
        return make_response('', 200)

    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image provided (field name "image")'}), 400

    if not _is_allowed_mime((file.mimetype or '').lower()):
        return jsonify({'error': 'Unsupported type (use PNG or JPEG)'}), 415

    image_bytes = file.read()
    try:
        resp = enterprise_agent.respond_to_image(image_bytes)
        return jsonify({'reply': resp}) if not isinstance(resp, (dict, list)) else jsonify(resp)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('CHAT_SERVER_PORT', 5001))
    app.run(host='0.0.0.0', port=port)
