import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# Load the trained model from the H5 file
model = tf.keras.models.load_model('roadSignModel_V2.h5')

# Define a function to preprocess the image
def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')  # Convert image to grayscale
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    return img_array


def get_sign_name():
    # Read the CSV file
    sign_names_df = pd.read_csv("classLables/label_names.csv")
    # Create a mapping between ClassId and SignName
    class_id_to_sign_name = dict(zip(sign_names_df["ClassId"], sign_names_df["SignName"]))
    return class_id_to_sign_name


# Define a route for the prediction service
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Preprocess the image
    img_file = request.files['image']
    img_path = 'temp_img.jpg'
    img_file.save(img_path)

    # Check if the file is a valid image
    if not os.path.isfile(img_path):
        return jsonify({'error': 'Invalid image file provided'}), 400

    try:
        # Preprocess the image
        processed_img = preprocess_image(img_path)

        # Make predictions
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions[0])

        # Convert the prediction to a human-readable format
        sign_labels = get_sign_name()
        predicted_sign = sign_labels[predicted_class]

        return jsonify({'prediction': predicted_sign}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary image file
        os.remove(img_path)

if __name__ == '__main__':
    app.run(debug=True, port=8083)