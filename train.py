from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Flatten, Dense, Dropout
# from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D,Activation
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.optimizers import Adam
from keras.applications.mobilenet_v1 import MobileNetV1
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,SeparableConv2D,BatchNormalization, Activation, Dense
# from tensorflow.python.keras.layers.b
import numpy as np

def MobileNET(weights_path=None):
    base_model = MobileNetV1(include_top=False, weights='imagenet', input_shape=(224,224,3))
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(102,activation='relu')(x)
    x = Dropout(0.25)(x)
    x=Dense(51,activation='relu')(x) 
    x = Dropout(0.25)(x)
    preds=Dense(num_class, activation='softmax')(x) #final layer with softmax activation

    model=Model(inputs=base_model.input,outputs=preds)
    model.summary()
