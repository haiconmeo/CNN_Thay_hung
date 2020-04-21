

import numpy as np
import cv2
from keras.models import Model, Sequential

from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import model_from_json
import matplotlib.pyplot as plt
from os import listdir

from keras.layers import Dense, Activation, Dropout, Flatten, Input,BatchNormalization, Convolution2D, ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D,Activation
from keras.layers import Conv2D, AveragePooling2D
from keras.applications.mobilenet import MobileNet
num_class=100
base_model = MobileNet(include_top=False, input_shape=(224,224,3))
x=base_model.output

x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x = Dropout(0.25)(x)
x=Dense(512,activation='relu')(x) 
x = Dropout(0.25)(x)
preds=Dense(num_class, activation='softmax')(x) 

model=Model(inputs=base_model.input,outputs=preds)
model.summary()

model.load_weights(".\models\model\model-005.hdf5")




face_cascade = cv2.CascadeClassifier(".\models\haarcascade_frontalface_default.xml")

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img








cap = cv2.VideoCapture(0) 

while(True):
	ret, img = cap.read()
	
	
	faces = face_cascade.detectMultiScale(img, 1.3, 5)
	
	for (x,y,w,h) in faces:
		if w > 130: 
			
			
			
			cv2.rectangle(img,(x,y),(x+w,y+h),(128,128,128),1) 
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] 
			
			try:
				
				margin = 30
				margin_x = int((w * margin)/100); margin_y = int((h * margin)/100)
				detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]
			except:
				print("detected face has no margin")
			
			try:
				
				detected_face = cv2.resize(detected_face, (224, 224))
				
				img_pixels = image.img_to_array(detected_face)
				img_pixels = np.expand_dims(img_pixels, axis = 0)
				img_pixels /= 255
				
				
				age_distributions = model.predict(img_pixels)
				apparent_age = str(np.argmax(age_distributions))
			
			
				
				info_box_color = (32, 84, 216 )
				
				triangle_cnt = np.array( [(x+int(w/2), y), (x+int(w/2)-20, y-20), (x+int(w/2)+20, y-20)] )
				cv2.drawContours(img, [triangle_cnt], 0, info_box_color, -1)
				cv2.rectangle(img,(x+int(w/2)-50,y-20),(x+int(w/2)+50,y-90),info_box_color,cv2.FILLED)
				
				cv2.putText(img, apparent_age, (x+int(w/2), y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)
				
			except Exception as e:
				print("exception",str(e))
			
	cv2.imshow('img',img)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break
	
	
cap.release()
cv2.destroyAllWindows()