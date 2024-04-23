## Import Libraries

# For data
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import tensorflow

# For Data Statistics
from matplotlib import pyplot as plt
## Import Keras objects for MLP Layers

from keras.models  import Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras.optimizers import SGD

import keras
import keras.utils

from numpy import mean
from numpy import std
from matplotlib import pyplot as plt

from keras.models import load_model

def imageTest(imgPath, modelPath):

  imgTest001= cv2.imread(imgPath)
  resize = cv2.resize(imgTest001, (100,100))
  #cv2.imshow("Something",resize)
  #cv2.waitKey(0)

  testModel= load_model(modelPath)

  val=testModel.predict(np.expand_dims(resize/255, 0))
  print(val)
  max_index= np.argmax(val, axis=None, out=None)

  match max_index:
      case 0:
        return("Weather is: Cloudy")
      case 1:
        return( "Weather is: Rain")
      case 2:
        return( "Weather is: Sunrise")
      case 3:
        return( "Weather is: Shine")
      case _:
        return( "Something went wrong.")