import base64
import io
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)

# Load saved keras model
MODEL_PATH = "mnist_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_data):
    # Decode base64 image string from canvas
    image_bytes = base64.b64decode(image_data.split(',')[1])
    img = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to Grayscale
    
    # Resize to 28x28 (MNIST input shape)
    img = img.resize((28, 28))
    
    # Normalize pixel values [0, 1]
    img_array = np.array(img) / 255.0
    
    # Reshape for neural network input
    if len(model.input_shape) == 4:
        img_array = img_array.reshape(1, 28, 28, 1)
    else:
        img_array = img_array.reshape(1, 28, 28)
        
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']
        
        # Preprocess and predict
        processed_img = preprocess_image(image_data)
        prediction_probs = model.predict(processed_img)
        predicted_digit = int(np.argmax(prediction_probs[0]))
        
        return jsonify({'prediction': predicted_digit})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)
