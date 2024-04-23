
import streamlit as st

import tensorflow as tf



@st.cache(allow_output_mutation=True)

def load_model():

  model=tf.keras.models.load_model('CNN_Model_7.h5')

  return model

model=load_model()

st.write("""

# Multi-Weather Classifier"""

)

file=st.file_uploader("Upload your Image",type=["jpg","png"])



import cv2

from PIL import Image,ImageOps

import numpy as np

def import_and_predict(image_data,model):

    size=(64,64)

    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)

    img=np.asarray(image)

    img_reshape=img[np.newaxis,...]

    prediction=model.predict(img_reshape)

    return prediction

if file is None:

    st.text("Please upload an image file")

else:

    image=Image.open(file)

    st.image(image,use_column_width=True)

    prediction=import_and_predict(image,model)

    class_names=['Sunrise', 'Shine', 'Rain', 'Cloudy']

    string="OUTPUT : "+class_names[np.argmax(prediction)]

    st.success(string)

