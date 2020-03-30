from tensorflow.python.keras.preprocessing import image
import os 
from tensorflow.python.keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
from train  import VGG_16
from tensorflow.python.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.python.keras.utils import  to_categorical
one_hot = 13
classname = ["1_15","16_20","21_25","26_30","31_35","36_40","41_45","46_50","51_55","56_60","61_65","66_70","70_"]

def load_file(file_path):
    img =  image.load_img(file_path, target_size=(224, 224))
    x =  image.img_to_array(img).reshape(1,-1)[0]

    return x

# print (load_file("./data/wiki_crop/00/23300_1962-06-19_2011.jpg"))
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
    if a >71:
        return classname[12]

def load_data():
    id =[]
    age = []
    x_train =[]
    folder = os.listdir('./data/wiki_crop')
    folder=["00"]
    for fo in folder :

        path = './data/wiki_crop/'+str(folder[0])
        # print (path)
        
        for i in os.listdir(path):
            # print (i)
            x =  i.split("_")
            
            a = int(x[2].split(".")[0])-int(x[1].split("-")[0])
            if a >3 and a <100:
                
                id.append(x[0])
                age.append(to_categorical(a,one_hot) )
                x_train.append(load_file(path+"/"+i))
    

train_x, test_x, train_y, test_y = load_data()
print (train_x.shape)
# train_x  = np.array(train_x)
# train_y  = np.array(train_y)
# model = VGG_16()
# epochs = 2
# batch_size = 199

# check_point = ModelCheckpoint(
#     filepath='models/classification_age_model.hdf5'
#     , monitor="val_loss"
#     , verbose=1
#     , save_best_only=True
#     , mode='auto'
# )

# Bat dau train
# for i in range(epochs):
#     print ("Train epoch: ", i)
#     ix_train = np.random.choice(train_x.shape[0], size=batch_size)
#     model.fit(
#         train_x[ix_train], train_y[ix_train]
#         , epochs=1
#         , validation_data=(test_x, test_y)
#         , callbacks=[check_point]
#     )
# model.fit(train_x,train_y)
# Luu model
model = load_model("models/classification_age_model.hdf5")
model.save_weights('models/age_model_weights.h5')

# # print (train_x[0])