#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import shutil
import glob
from tqdm import tqdm


# In[36]:


Raw_DIR= 'C://Users//lalga//Downloads//mrlEyes_2018_01//mrlEyes_2018_01'
for dirpath, dirname, filenames in os.walk(Raw_DIR):
    for i in tqdm([f for f in filenames if f.endswith('.png')]):
        if i.split('_')[4]=='0':
            shutil.copy(src=dirpath+'/'+i, dst='C://Users//lalga//Downloads//mrlEyes_2018_01//Prepared_Data//Close Eyes')
        
        elif i.split('_')[4]=='1':
            shutil.copy(src=dirpath+'/'+i, dst='C://Users//lalga//Downloads//mrlEyes_2018_01//Prepared_Data//Open Eyes')


# In[17]:


import os
import shutil

# Path to the folder containing the images
src_folder = 'C://Users//lalga//Downloads//mrlEyes_2018_01//Prepared_Data//Open Eyes'
# Path to the folder where you want to move the images
dest_folder = 'C://Users//lalga//Downloads//mrlEyes_2018_01//Prepared_Data//train//Open Eyes'

# Get a list of all the image files in the source folder
image_files = [os.path.join(src_folder, f) for f in os.listdir(src_folder) if f.endswith('.png')]

# Calculate the number of images to move
num_images = len(image_files)
print(num_images)
#num_images_to_move = num_images * 

# Loop through the first half of the image files and move them to the destination folder
#for i in range(num_images_to_move):
    #image_file = image_files[i]
    #shutil.move(image_file, dest_folder)


# 41946 - close eyes
# 42952 - open eyes

# In[40]:


import os
import shutil

# Path to the folder containing the images
src_folder = 'C://Users//lalga//Downloads//mrlEyes_2018_01//Prepared_Data//Close Eyes'
# Path to the folder where you want to move the images
dest_folder = 'C://Users//lalga//Downloads//mrlEyes_2018_01//Prepared_Data//test//Close Eyes'

# Get a list of all the image files in the source folder
image_files = [os.path.join(src_folder, f) for f in os.listdir(src_folder) if f.endswith('.png')]





# Loop through the images to move and move them to the destination folder
for image_file in image_files[31460:]:
    shutil.copy(image_file, dest_folder)
        


# In[41]:


import os
import shutil

# Path to the folder containing the images
src_folder = 'C://Users//lalga//Downloads//mrlEyes_2018_01//Prepared_Data//Open Eyes'
# Path to the folder where you want to move the images
dest_folder = 'C://Users//lalga//Downloads//mrlEyes_2018_01//Prepared_Data//test//Open Eyes'

# Get a list of all the image files in the source folder
image_files = [os.path.join(src_folder, f) for f in os.listdir(src_folder) if f.endswith('.png')]



# Loop through the images to move and move them to the destination folder
for image_file in image_files[32214:]:
    shutil.copy(image_file, dest_folder)


# In[2]:


import os
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout,Input,Flatten,Dense,MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[3]:


BATCH_SIZE = 4


# In[4]:


train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range = 0.2,shear_range = 0.2,
    zoom_range = 0.2,width_shift_range = 0.2,
    height_shift_range = 0.2, validation_split = 0.2)


# In[5]:


train_data= train_datagen.flow_from_directory(os.path.join('C://Users//lalga//Downloads//mrlEyes_2018_01//Prepared_Data//train'),
                                target_size = (80,80), batch_size = BATCH_SIZE, 
                                class_mode = 'categorical',subset='training' )


# In[6]:


validation_data= train_datagen.flow_from_directory(os.path.join('C://Users//lalga//Downloads//mrlEyes_2018_01//Prepared_Data//train'),
                                target_size = (80,80), batch_size = BATCH_SIZE, 
                                class_mode = 'categorical', subset='validation')


# In[7]:


test_datagen = ImageDataGenerator(rescale = 1./255)


# In[8]:


test_data = test_datagen.flow_from_directory(os.path.join('C://Users//lalga//Downloads//mrlEyes_2018_01//Prepared_Data//test'),
                                target_size=(80,80), batch_size = BATCH_SIZE, class_mode='categorical')


# In[9]:


bmodel = InceptionV3(include_top = False, weights = 'imagenet', 
                     input_tensor = Input(shape = (80,80,3)))
nmodel = bmodel.output
nmodel = Flatten()(nmodel)
nmodel = Dense(64, activation = 'relu')(nmodel)
nmodel = Dropout(0.5)(nmodel)
nmodel = Dense(2,activation = 'softmax')(nmodel)


# In[10]:


model = Model(inputs = bmodel.input, outputs= nmodel)
for layer in bmodel.layers:
    layer.trainable = False


# In[11]:


model.compile(optimizer = 'Adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])


model.fit(train_data,steps_per_epoch = train_data.samples// BATCH_SIZE,
                   validation_data = validation_data,
                   validation_steps = validation_data.samples// BATCH_SIZE,
                    epochs = 5)


# In[12]:


# Save the final model
model.save('mrleye_detection_inceptionv3_model_final.h5')


# 4D model
# 

# In[13]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation

model = Sequential()

# Convolutional layers
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(100, 100, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(384, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(1024, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully-connected layers
model.add(Flatten())
model.add(Dense(16384))
model.add(Activation('relu'))

model.add(Dense(180))
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))



# In[ ]:


from keras.optimizers import Adam

# create data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=0.2, shear_range=0.2,
                                   zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, 
                                   validation_split=0.2)
train_data = train_datagen.flow_from_directory(os.path.join('C://Users//lalga//Downloads//mrlEyes_2018_01//Prepared_Data//train'),
                                               target_size=(80, 80), batch_size=BATCH_SIZE, 
                                               class_mode='categorical', subset='training')
validation_data = train_datagen.flow_from_directory(os.path.join('C://Users//lalga//Downloads//mrlEyes_2018_01//Prepared_Data//train'),
                                                    target_size=(80, 80), batch_size=BATCH_SIZE, 
                                                    class_mode='categorical', subset='validation')
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(os.path.join('C://Users//lalga//Downloads//mrlEyes_2018_01//Prepared_Data//test'),
                                             target_size=(80, 80), batch_size=BATCH_SIZE, class_mode='categorical')

# compile model
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# fit model
history = model.fit_generator(train_data, steps_per_epoch=train_data.samples//32, epochs=10, 
                              validation_data=validation_data, validation_steps=validation_data.samples//32)


# In[ ]:





# In[ ]:





# In[58]:


# Save the final model
model.save('mrleye_detection_model_final.h5')


# In[1]:


import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer


# In[2]:


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
model = load_model('mrleye_detection_model_final.h5')


# In[6]:


mixer.init()
sound= mixer.Sound('mixkit-classic-alarm-995.wav')
cap = cv2.VideoCapture(0)
Score = 0
while True:
    ret, frame = cap.read()
    height,width = frame.shape[0:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray, scaleFactor= 1.2, minNeighbors=3)
    eyes= eye_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors=1)
    
    cv2.rectangle(frame, (0,height-50),(200,height),(0,0,0),thickness=cv2.FILLED)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h), color= (255,0,0), thickness=3 )
        
    for (ex,ey,ew,eh) in eyes:
        #cv2.rectangle(frame,pt1=(ex,ey),pt2=(ex+ew,ey+eh), color= (255,0,0), thickness=3 )
        
        # preprocessing steps
        eye= frame[ey:ey+eh,ex:ex+ew]
        eye= cv2.resize(eye,(80,80))
        eye= eye/255
        eye= eye.reshape(80,80,3)
        eye= np.expand_dims(eye,axis=0)
        # preprocessing is done now model prediction
        prediction = model.predict(eye)
        print(prediction)
        print(prediction[0][1])
        print(prediction[0][0])
        # if eyes are closed
        if prediction[0][0]>0.90:
            cv2.putText(frame,'closed',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)
            cv2.putText(frame,'Score'+str(Score),(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)
            Score=Score+1
            if(Score>15):
                try:
                    sound.play()
                except:
                    pass
            
        # if eyes are open
        elif prediction[0][1]>0.90:
            cv2.putText(frame,'open',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)      
            cv2.putText(frame,'Score'+str(Score),(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)
            Score = Score-1
            if (Score<0):
                Score=0
            
        
    cv2.imshow('frame',frame)
    if cv2.waitKey(33) & 0xFF==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




