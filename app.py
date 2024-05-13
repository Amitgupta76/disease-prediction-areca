from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import os 
import numpy as np
import cv2
from werkzeug.utils import secure_filename
from chatbot import chat_loop

app = Flask(__name__)

# Load the trained model
def load_model():
    global model
    model = tf.keras.models.load_model('areca_model_1')

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
        result = chat_loop(f"You are an expert in plant pathology, specific to areca nut or leaf diseases. Your task is to recommend a solution or a medicine for {predicted_disease} in areca leaf/nut. Avoid technical jargon and explain it in the simplest of words. Your solution must be a heading with short and crisp sub-points and must not exceed 400 words.")  # Call chat_loop function
        
        return render_template('result.html', disease=predicted_disease, result=result)
    else:
        return render_template('index.html', message='Something went wrong')
    
# Define route for result page with chatbot
@app.route('/chat', methods=['POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form['user_input']
        bot_response = chat_loop(f'You are an expert in plants and plant pathology, specific to areca nut or leaf diseases and must not answer anything else, if the question is out of scope, please provide the answer as "Question out of scope". Your task is to recommend a solution for a question, question: {user_input}. Avoid technical jargon and explain it in the simplest of words. Your solution must be a heading with short and crisp sub-points and must not exceed 100 words.')  # Call chat_loop function
        return jsonify({'bot_response': bot_response})
    else:
        return jsonify({'error': 'Method not allowed'})

if __name__ == '__main__':
    load_model()  # Load the model when the Flask app starts
    app.run(debug=True)
