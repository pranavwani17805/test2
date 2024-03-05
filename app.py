from flask import Flask, render_template, request
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import secrets

app = Flask(__name__)

# Define classes
classes = ['plant', 'not_plant']

# Load the pre-trained model
#new_model = tf.keras.models.load_model('1plant.h5')
new_model=tf.keras.models.load_model('2plant.h5')
# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for handling image classification
@app.route('/classify', methods=['POST'])
def classify():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return render_template('result.html', result='Error', message='No image uploaded')

    # Get the uploaded image
    uploaded_image = request.files['image']

    # Save the uploaded image temporarily
    filename = secrets.token_hex(8) + '.jpg'
    file_path = os.path.join('uploads', filename)
    uploaded_image.save(file_path)

    # Read the image using PIL
    img = Image.open(file_path)
    img = img.resize((224, 224))

    # Preprocess the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Predict the class probabilities
    probs = new_model.predict(x)
    max_prob = max(probs[0])

    if max_prob > 0.7:
        predicted_class = np.argmax(probs, axis=1)
        result = classes[predicted_class[0]]
        message = 'Plant identified'
    else:
        result = 'Plant not identified'
        message = 'Probability is less than 70%'

    # Calculate accuracy
    #accuracy = max_prob * 100
    accuracy=round(max_prob*100,2)
    # Pass the image URL, result, message, and accuracy to the result.html template
    return render_template('result.html', result=result, message=message, accuracy=accuracy, image_url=file_path)

if __name__ == '__main__':
    app.run(debug=True)
