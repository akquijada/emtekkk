from flask import Flask, render_template, request
import numpy as np
import cv2 as cv
from tensorflow.keras import models

app = Flask(__name__, template_folder='C:/Users/tipqc/Desktop/DELETE AFTER/')

# Load the pre-trained model
model = models.load_model('CNN_Model_7.h5')

# Class names for CIFAR-10
class_names = ['Rain', 'Cloudy', 'Sunrise', 'Shine']


def preprocess_image(img_path):
    img = cv.imread(img_path)
    img = cv.resize(img, (100, 100))
    img = img / 255.0
    img = img[None, :]
    return img


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle the uploaded image
        file = request.files['file']
        if file:
            # Save the uploaded image
            img_path = 'uploads/uploaded_image.png'
            file.save(img_path)

            # Preprocess the image
            img = preprocess_image(img_path)

            # Make a prediction
            prediction = model.predict(img)
            index = np.argmax(prediction)
            result = class_names[index]

            return render_template("index.html", result=result, image_path=img_path)

    return render_template("index.html", result=None, image_path=None)


if __name__ == '__main__':
    app.run(debug=True)