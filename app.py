from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import numpy as np
from PIL import Image, ImageOps
import requests
import tensorflow as tf
import warnings
import openai
import google.generativeai as genai
import pathlib
import os
from datetime import datetime


api_key = "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"
gemini_key = "AIzaSyBBH-xlLo8IeAyQoIKMMmkyVAHPze9nMvU"
OPENAI_API_KEY=api_key
openai.api_key=OPENAI_API_KEY
genai.configure(api_key=gemini_key)
# Ignore warnings
warnings.filterwarnings("ignore")

# Initialize Flask application
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)



# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
print("Model loaded")

# Assuming labels are loaded similarly as before
class_names = [line.strip() for line in open("labels.txt")]

save_dir = 'generated_images'
os.makedirs(save_dir, exist_ok=True)

# Function to preprocess image
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array.reshape((1, 224, 224, 3))

@app.route("/")
def sayHello():
    return  jsonify({"message" : "hello"})

# Route to predict endpoint
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['image']

    try:
        image = Image.open(file).convert("RGB")
        processed_image = preprocess_image(image)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], processed_image)

        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = np.argmax(prediction)
        class_name = class_names[predicted_index]
        confidence_score = float(prediction[0][predicted_index])

        if "uncut" in class_name:
            save_path = "image.jpg"

            # Save the image locally
            image.save(save_path)
            # Access your API key as an environment variable.
            genai.configure(api_key=gemini_key)
            # Choose a model that's appropriate for your use case.
            model = genai.GenerativeModel('gemini-1.5-flash')

            image1 = {
                'mime_type': 'image/jpeg',
                'data': pathlib.Path(save_path).read_bytes()
            }

            prompt = "What are the colors of this gemstone"

            response = model.generate_content([prompt, image1])
            print(response.text)
            os.remove(save_path)

            return jsonify({
                'class_name': class_name,
                'confidence_score': confidence_score,
                "response" : response.text
            })
        else:
            return jsonify({
                'class_name': class_name,
                'confidence_score': confidence_score
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    # Prepare request headers with API key
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    # Prepare request data
    api_data = {
        'prompt': prompt,
        'num_images': 2,
        'model': 'dall-e-2',
        'response_format' : 'url'
    }

    # Send POST request to DALL·E API
    response = requests.post('https://api.openai.com/v1/images/generations', json=api_data, headers=headers)

    # Check if request was successful
    if response.status_code == 200:
        # Extract and display the generated image
        response_data = response.json()
        image_urls = [image['url'] for image in response_data['data']]


        downloaded_images = []
        for url in image_urls:
            image_response = requests.get(url)
            if image_response.status_code == 200:
                # Create a filename with the current timestamp
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
                image_filename = os.path.join(save_dir, f'{timestamp}.png')

                # Save the image to disk
                with open(image_filename, 'wb') as image_file:
                    image_file.write(image_response.content)

                downloaded_images.append(f'/images/{os.path.basename(image_filename)}')
        return jsonify({'images': downloaded_images})
    else:
        return jsonify({'error': f"Error {response.status_code}: {response.text}"}), response.status_code

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(save_dir, filename)


@app.route("/")
def test():
    return jsonify({"message" : "hello"})

if __name__ == '__main__':
    app.run(debug=True)
