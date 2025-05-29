from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
from datetime import datetime

app = Flask(__name__)

# Load or train the MNIST model
def load_or_train_model():
    model_path = 'mnist_model.h5'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

        # Define the model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # Compile and train the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Training model... This may take a few minutes.")
        model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1)
        model.save(model_path)
    return model

model = load_or_train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Process the image
        img = Image.open(file).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28
        img_array = np.array(img)
        # Apply improved thresholding and centering
        thresh = img_array > img_array.mean()  # Adaptive threshold based on image mean
        img_array = thresh.astype('float32') * 255
        img_array = img_array.reshape(1, 28, 28, 1) / 255.0

        # Make prediction
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        return jsonify({
            'digit': int(predicted_digit),
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    digit = data.get('correctDigit')
    with open('feedback.json', 'a') as f:
        json.dump({'digit': digit, 'timestamp': str(datetime.now())}, f)
        f.write('\n')
    return jsonify({'message': 'Feedback received'})

@app.route('/get_feedback', methods=['GET'])
def get_feedback():
    try:
        feedback_list = []
        if os.path.exists('feedback.json'):
            with open('feedback.json', 'r') as f:
                for line in f:
                    if line.strip():
                        feedback_list.append(json.loads(line))
        return jsonify(feedback_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
