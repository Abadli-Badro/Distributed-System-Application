from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import tensorflow as tf
import numpy as np
import cv2
import threading
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static'

# Load pre-trained TensorFlow model (ensure this is thread-safe)
model = tf.keras.models.load_model('ressources/SignLanguage.h5')

# Create a thread pool for handling requests
executor = ThreadPoolExecutor(max_workers=4)  # Adjust max_workers based on your server's capacity

# Mapping from class index to alphabet
alphabet_map = {i: chr(65 + i) for i in range(26)}  # {0: 'A', 1: 'B', ..., 25: 'Z'}

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        image = cv2.resize(image, (28, 28))  # Resize to 28x28 pixels
        image = np.expand_dims(image, axis=-1)  # Add channel dimension (28, 28) -> (28, 28, 1)
        image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 28, 28, 1)
        return image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def classify_image(file_path):
    img_array = preprocess_image(file_path)
    if img_array is None:
        return "Error: Unable to preprocess image"
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return alphabet_map.get(predicted_class, "Unknown")

@app.route('/', methods=['GET', 'POST'])
def home():
    print(f"Processing on Thread ID: {threading.get_ident()}")
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)

        # Use threading to handle prediction in parallel
        future = executor.submit(classify_image, file_path)
        predicted_alphabet = future.result()

        # Pass result to the classify.html template
        return render_template("classify.html", result=predicted_alphabet, img_path=file_path)

    return render_template('home.html', form=form)

if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, threaded=True)
