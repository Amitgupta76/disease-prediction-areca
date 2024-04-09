from flask import Flask, render_template, request
import tensorflow as tf
import os 
import numpy as np
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
def load_model():
    global model
    model = tf.keras.models.load_model('areca_cnn_model_2')

# Function to process the uploaded image
def process_image(image1):
    image = cv2.imread(image1)
    image = cv2.resize(image, (150, 150))
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)
    return image

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', message='No image file selected')
    
    f = request.files['image']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    
    if f.filename == '':
        return render_template('index.html', message='No image file selected')
    
    if f:
        # Process the image
        img_array = process_image(file_path)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        class_names = ['stem cracking', 'Stem_bleeding', 'Healthy_Leaf', 'yellow leaf disease', 'healthy_foot', 'Healthy_Trunk', 'Mahali_Koleroga', 'bud borer', 'Healthy_Nut']
        predicted_disease = class_names[predicted_class]
        
        return render_template('result.html', disease=predicted_disease)
    else:
        return render_template('index.html', message='Something went wrong')

if __name__ == '__main__':
    load_model()  # Load the model when the Flask app starts
    app.run(debug=True)
