from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load or train the MNIST model
def load_or_train_model():
    model_path = 'mnist_model.h5'
    try:
        if os.path.exists(model_path):
            logger.info("Loading existing model from %s", model_path)
            model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.info("Model not found, training a new model...")
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            logger.info("Training model... This may take a few minutes.")
            model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=1)
            model.save(model_path)
            logger.info("Model trained and saved as %s", model_path)
        return model
    except Exception as e:
        logger.error("Failed to load or train model: %s", str(e))
        return None

model = load_or_train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logger.error("No file uploaded in request")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("No file selected")
        return jsonify({'error': 'No file selected'}), 400

    try:
        file.seek(0)  # Ensure file pointer is at the start
        img = Image.open(io.BytesIO(file.read())).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28
        img_array = np.array(img)
        if img_array.size == 0:
            logger.error("Image array is empty after processing")
            return jsonify({'error': 'Invalid image'}), 400
        
        # Simplify thresholding to avoid potential issues with noisy images
        img_array = (img_array > img_array.mean()).astype('float32') * 255
        img_array = img_array.reshape(1, 28, 28, 1) / 255.0

        if model is None:
            logger.error("Model is not loaded")
            return jsonify({'error': 'Model not loaded'}), 500

        logger.info("Making prediction...")
        prediction = model.predict(img_array, verbose=0)
        predicted_digit = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        logger.info("Prediction successful: digit=%d, confidence=%.2f", predicted_digit, confidence)

        return jsonify({
            'digit': int(predicted_digit),
            'confidence': confidence
        })
    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        if not data or 'correctDigit' not in data:
            logger.error("Invalid feedback data: %s", data)
            return jsonify({'error': 'Invalid feedback data'}), 400
        
        digit = data.get('correctDigit')
        with open('feedback.json', 'a') as f:
            json.dump({'digit': digit, 'timestamp': str(datetime.now())}, f)
            f.write('\n')
        logger.info("Feedback recorded: digit=%s", digit)
        return jsonify({'message': 'Feedback received'})
    except Exception as e:
        logger.error("Feedback error: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/get_feedback', methods=['GET'])
def get_feedback():
    try:
        feedback_list = []
        if os.path.exists('feedback.json'):
            with open('feedback.json', 'r') as f:
                for line in f:
                    if line.strip():
                        feedback_list.append(json.loads(line))
        logger.info("Retrieved feedback: %d entries", len(feedback_list))
        return jsonify(feedback_list)
    except Exception as e:
        logger.error("Get feedback error: %s", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
