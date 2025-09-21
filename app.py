from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
import io
from PIL import Image
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MODEL_PATH = 'my_image_classifier.h5'
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variable to store the model
model = None

def load_model():
    """Load the trained model"""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            return False, f"Model file not found at '{MODEL_PATH}'"
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return True, "Model loaded successfully"
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
    return True, "Model already loaded"

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_data):
    """Preprocess image for model prediction"""
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model's expected input size (32x32 for CIFAR-10)
        img = img.resize((32, 32), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize pixel values (0-1)
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, None
    except Exception as e:
        return None, f"Error preprocessing image: {str(e)}"

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Load model if not already loaded
        success, message = load_model()
        if not success:
            return jsonify({
                'success': False,
                'error': message
            }), 500

        # Check if file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400

        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload PNG, JPG, JPEG, GIF, BMP, or WEBP files.'
            }), 400

        # Read and preprocess the image
        image_data = file.read()
        processed_image, error = preprocess_image(image_data)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400

        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get the predicted class and confidence
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions[0]) * 100)
        
        # Get all predictions for additional info
        all_predictions = []
        for i, class_name in enumerate(CLASS_NAMES):
            all_predictions.append({
                'class': class_name,
                'confidence': float(predictions[0][i] * 100)
            })
        
        # Sort predictions by confidence
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions[:3]  # Top 3 predictions
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred during prediction: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    success, message = load_model()
    return jsonify({
        'status': 'healthy' if success else 'error',
        'model_status': message
    })

if __name__ == '__main__':
    print("Starting Image Recognition Web App...")
    print("Make sure your model file 'my_image_classifier.h5' is in the same directory!")
    app.run(debug=True, host='0.0.0.0', port=5000)
