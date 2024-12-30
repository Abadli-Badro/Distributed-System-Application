from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import tensorflow as tf
import numpy as np
import cv2 

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static'

# Loading pre-trained TensorFlow model
model = tf.keras.models.load_model('ressources\\SignLanguage.h5')

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

alphabet_map = {i: chr(65 + i) for i in range(26)}  # {0: 'A', 1: 'B', ..., 25: 'Z'}

def preprocess_image(image_path):
    """
    Preprocess an image to match the MNIST dataset format.
    Args:
        image_path (str): Path to the image.
    Returns:
        np.array: Preprocessed image ready for prediction.
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    image = cv2.resize(image, (28, 28))  # Resize to 28x28 pixels
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (28, 28) -> (28, 28, 1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 28, 28, 1)
    return image

@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)

        # Prepare the image and perform prediction
        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]  # Get the predicted class index
        predicted_alphabet = alphabet_map[predicted_class]

        # Pass result to the classify.html template
        return render_template("classify.html", result=predicted_alphabet , img_path = file_path)

    return render_template('home.html', form=form)

if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)

