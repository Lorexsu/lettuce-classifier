from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)

# Load model - handle different paths for local vs deployed
model_path = os.getenv("MODEL_PATH", "lettuce_project/runs/detect/lettuce_new_new/weights/best.pt")

try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading model from {model_path}: {e}")
    print("Using default YOLOv8 model as fallback")
    model = YOLO("yolov8n.pt")

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.json
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        results = model.predict(image, conf=0.5)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = results[0].names[cls_id]
            
            return jsonify({
                'detected': True,
                'classification': label,
                'confidence': conf
            })
        else:
            return jsonify({
                'detected': False
            })
    except Exception as e:
        return jsonify({
            'detected': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'Lettuce Classification API',
        'status': 'running'
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
