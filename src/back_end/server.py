from flask import Flask, request, jsonify, make_response, send_from_directory
try:
    from flask_cors import CORS
except Exception:
    CORS = None
import os
import uuid
from werkzeug.utils import secure_filename

# Import the helper from chat_cli
from agents.base_agent import Agent

app = Flask(__name__)
# Enable CORS if flask_cors is available. This allows the frontend (served on another port)
# to send POST requests and the browser preflight (OPTIONS) to succeed.
if CORS:
    CORS(app, resources={r"/api/*": {"origins": "*"}})

civilian_agent = Agent(civilian=1)
enterprise_agent = Agent(civilian=0)

# --- Image upload config ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# If flask_cors is not installed, add a simple after_request hook and ensure
# each route accepts OPTIONS so preflight requests succeed.
@app.after_request
def add_cors_headers(response):
    # Allow any origin (development). Narrow this in production.
    response.headers.setdefault('Access-Control-Allow-Origin', '*')
    response.headers.setdefault('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.headers.setdefault('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    return response

# Serve uploaded files
@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, conditional=True)

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

@app.route('/api/chat/civillian_images', methods=['POST', 'OPTIONS'])
def upload_images():
    if request.method == 'OPTIONS':
        return make_response('', 200)

    file = request.files.get('image')
    if not file:
        return jsonify({'error':'No images provided (field name "images")'}), 400

    image_bytes = file.read()

    if not file.mimetype in ('image/png', 'image/jpeg'):
        return {'error': 'Unsupported type'}, 415
    
    response = civilian_agent.respond_to_image(image_bytes)

    return response


@app.route('/api/chat/enterprise_images', methods=['POST', 'OPTIONS'])
def upload_images():
    if request.method == 'OPTIONS':
        return make_response('', 200)

    file = request.files.get('image')
    if not file:
        return jsonify({'error':'No images provided (field name "images")'}), 400

    image_bytes = file.read()

    if not file.mimetype in ('image/png', 'image/jpeg'):
        return {'error': 'Unsupported type'}, 415
    
    response = enterprise_agent.respond_to_image(image_bytes)

    return response


if __name__ == '__main__':
    port = int(os.getenv('CHAT_SERVER_PORT', 5001))
    app.run(host='0.0.0.0', port=port)
