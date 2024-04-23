from flask import Flask, request, jsonify
import tensorflow as tf  # If you're using TensorFlow
import sample001 as samp1



app = Flask(__name__)

# Load your model
# For TensorFlow/Keras
model = tf.keras.models.load_model("C:/Users/tipqc/Desktop/DELETE AFTER/CNN_Model_7.h5")
# For PyTorch
# model = torch.load('model.pth')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    result=samp1.imageTest("C:/Users/tipqc/Desktop/DELETE AFTER/rainy-weather-1.jpg",
          "C:/Users/tipqc/Desktop/DELETE AFTER/CNN_Model_7.h5")
    
    return jsonify(str(result))


if __name__ == '__main__':
    app.run(debug=True)