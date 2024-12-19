from flask import Flask, request, jsonify, render_template_string
import os
from PIL import Image
from io import BytesIO
import base64
from datetime import datetime
from tensorflow.keras.models import load_model
import numpy as np
import requests  # Import requests for Pushover API

app = Flask(__name__)

# Folders for storing images (relative paths)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static/uploads/')
HISTORY_FOLDER = os.path.join(os.getcwd(), 'static/history/')
MODEL_PATH = os.path.join(os.getcwd(), 'face_classification_model.keras')

# Ensure the directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HISTORY_FOLDER, exist_ok=True)

# Global variables to store state
latest_result = None
latest_image_path = None
intrusion_history = []  # Stores tuples of (timestamp, classification, image_path)

# Load pre-trained model (for classification)
model = load_model(MODEL_PATH)
CLASS_NAMES = ['Aryan','Tanisha','Garvit']

# Pushover API credentials
PUSHOVER_API_URL = 'https://api.pushover.net:443/1/messages.json'
USER_KEY = 'uuwcqi534ixqgohoysjw9mqj7sro2s'  # Replace with your Pushover user key
API_TOKEN = 'admkekuo3qshgofecb45n4ji8gng6t'  # Replace with your Pushover API token

# HTML content embedded in the Python script
# Routes
@app.route('/')
def index():
    """Serve the dashboard."""
    with open("index.html") as f:
        return render_template_string(f.read())

@app.route('/upload', methods=['POST'])
def upload():
    global latest_result, latest_image_path, intrusion_history

    try:
        # Parse JSON payload
        data = request.get_json()
        if not data or 'image' not in data or 'timestamp' not in data:
            return jsonify({"error": "Invalid data format"}), 400

        # Extract Base64-encoded image and timestamp
        base64_image = data['image']
        timestamp = data['timestamp']

        # Decode the Base64 image
        image_data = base64.b64decode(base64_image)
        img = Image.open(BytesIO(image_data))

        # Save the image as capture.jpg (overwrites the previous file)
        latest_image_path = os.path.join(UPLOAD_FOLDER, 'capture.jpg')
        img.save(latest_image_path)

        # Save the image in the history folder with a timestamp-based filename
        timestamp_for_filename = timestamp.replace(":", "-")
        history_image_path = os.path.join(HISTORY_FOLDER, f'history_{timestamp_for_filename}.jpg')
        img.save(history_image_path)

        # Classify the image
        classification_result = classify_image(latest_image_path)
        latest_result = classification_result["classification"]

        # Add to history
        intrusion_history.append((timestamp, latest_result, history_image_path))

        # Send Pushover alert if the result is "INTRUDER"
        if latest_result == "Intruder":
            send_pushover_notification("Intruder detected! Check your surveillance system.")

        # Return classification result with confidence and timestamp
        return jsonify({
            "result": classification_result["classification"],
            "confidence": classification_result["confidence"],
            "timestamp": timestamp
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/result', methods=['GET'])
def get_result():
    # Return the latest classification result
    result = latest_result if latest_result else "No classification result available."
    image_path = f"static/uploads/{os.path.basename(latest_image_path)}" if latest_image_path else None
    return jsonify({"result": result, "image_path": image_path}), 200


@app.route('/history', methods=['GET'])
def get_history():
    if not intrusion_history:
        return jsonify({"history": []}), 200

    # Return all past intrusion timestamps, classifications, and image paths
    return jsonify({
        "history": [
            {"timestamp": ts, "classification": cls, "image_path": f"static/history/{os.path.basename(img_path)}"}
            for ts, cls, img_path in intrusion_history
        ]
    }), 200


def classify_image(image_path):
    # Preprocess the image
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Adjust based on your model input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array.reshape(1, 224, 224, 3)  # Reshape for model input

    # Make prediction
    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100  # Get the highest confidence in percentage
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    # Determine the classification based on confidence
    if confidence < 75:
        classification = "Intruder"
    else:
        classification = CLASS_NAMES[predicted_class_index]

    print(f"Predicted class: {classification} with confidence: {confidence:.2f}%")
    return {"classification": classification, "confidence": confidence}


def send_pushover_notification(message):
    """Send a Pushover notification"""
    payload = {
        'user': USER_KEY,
        'token': API_TOKEN,
        'message': message,
    }
    response = requests.post(PUSHOVER_API_URL, data=payload)
    return response.json()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)