from tensorflow.python.keras.preprocessing import image
import os 
from tensorflow.python.keras.models import load_model

import numpy as np
import cv2

from tensorflow.python.keras.utils import  to_categorical
one_hot = 13
classname = ["1_15","16_20","21_25","26_30","31_35","36_40","41_45","46_50","51_55","56_60","61_65","66_70","70_"]
parent_dir = "train"
def makedir(classname):
    os.makedirs(parent_dir)
    for i in classname:
        path = os.path.join(parent_dir, i) 
        os.makedirs(path) 
        print("Directory '%s' created" %i)
def checkage(a):
    
    if a <16:
        return classname[0]
    if a <21:
        return classname[1]
    if a <26:
        return classname[2]
    if a <31:
        return classname[3]
    if a <36:
        return classname[4]
    if a <41:
        return classname[5]
    if a <46:
        return classname[6]
    if a <51:
        return classname[7]
    if a <56:
        return classname[8]
    if a <61:
        return classname[9]
    if a <66:
        return classname[10]
    if a <71:
        return classname[11]
    if a >70:
        return classname[12]
    

def load_img(x):

    anh  = cv2.imread(x)
    anh = cv2.resize(anh,(244,244))
    face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face.detectMultiScale(anh, 1.2, 5)

    if len(faces)==0:
        return None
    x, y, w, h = faces[0]
    anh_2 = cv2.rectangle(anh, (x, y), (x + w, y + h), (128, 128, 128), 1)
    detected_face = anh_2[int(y):int(y + h), int(x):int(x + w)]
    # detected_face = cv2.resize(detected_face, (224, 224))
    
    return detected_face

def load_data():
    
    folder = os.listdir("./data/wiki_crop")
    folder_2 = ["90","91","92","93","94","95","96","97","98","99"]
    for fo in folder_2 :
      
        path = "./data/wiki_crop/"+str(fo)
        print (fo+":")
        dem =0
        for i in os.listdir(path):
            dem+=1

            print ("nap :",path+"/"+i)
            tam = load_img(path+"/"+i)
            if tam is None :
                continue
            x =  i.split("_")
          
            a = int(x[2].split(".")[0])-int(x[1].split("-")[0])
            
            if a >3 and a <100:
                class_age = checkage(a)
                print (a)
                _path = os.path.join(parent_dir, class_age)
                id=x[0]+".jpg"
                path_img  = os.path.join(_path,id)
                # print(path_img)  
              
                cv2.imwrite(path_img,tam)
          
load_data()