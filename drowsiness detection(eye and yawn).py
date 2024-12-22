#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install cv2')


# In[10]:


import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report
from pygame import mixer


# In[11]:


from keras.models import load_model
# Load the trained model
modelyawn = load_model('yawn_detection_model2.h5')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
modelmrl = load_model('mrleye_detection_inceptionv3_model_final.h5')


# In[14]:


# Load the face detector cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the mouth detector cascade classifier
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

# Initialize the video capture object
mixer.init()
sound= mixer.Sound('mixkit-classic-short-alarm-993.wav')

Score = 0
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FPS, 30)  

def detect_eyes(frame,faces):
    height, width = frame.shape[:2]
    Score = 0
    
    # Draw a black rectangle at the bottom of the frame for displaying text
    cv2.rectangle(frame, (0, height-75), (200, height), (0, 0, 0), thickness=cv2.FILLED)
    
    # Detect faces and draw a blue rectangle around them
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=3)
        
        # Detect eyes inside the face region
        eyes = cv2.CascadeClassifier('haarcascade_eye.xml').detectMultiScale(frame[y:y+h, x:x+w], 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            # Uncomment the following line to draw a rectangle around the eyes
            #cv2.rectangle(frame, pt1=(x+ex, y+ey), pt2=(x+ex+ew, y+ey+eh), color=(255, 0, 0), thickness=3)
            
            # Preprocess the eye image for the model
            eye = frame[y+ey:y+ey+eh, x+ex:x+ex+w]
            eye = cv2.resize(eye, (80, 80))
            eye = eye / 255
            eye = eye.reshape(80, 80, 3)
            eye = np.expand_dims(eye, axis=0)
            
            # Make a prediction with the model
            prediction = modelmrl.predict(eye)
            
            # If eyes are closed, increase the score and play a sound if the score is high enough
            if prediction[0][0] > 0.8:
                y='closed'
                cv2.putText(frame, 'closed', (10, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                
            
            # If eyes are open, decrease the score
            elif prediction[0][1] > 0.50:
                y='open'
                cv2.putText(frame, 'open', (10, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)      
                
            
            return y



def yawndetect(frame,faces):
    img_height, img_width = frame.shape[:2]

    
# For each detected face, detect mouths and classify yawning
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = frame[y:y+h, x:x+w]

        # Convert the face ROI to grayscale for mouth detection
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Detect mouths in the grayscale face ROI
        mouths = mouth_cascade.detectMultiScale(gray_roi)

        # For each detected mouth, classify yawning
        for (mx, my, mw, mh) in mouths:
            # Extract the mouth ROI
            mouth_roi = gray_roi[my:my+mh, mx:mx+mw]
            
            # Convert the mouth ROI to a color image with three channels
            mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_GRAY2BGR)

            # Resize the mouth ROI to match the input size of the model
            mouth_roi = cv2.resize(mouth_roi, (img_height, img_width))

            # Resize the mouth ROI to match the input size of the model
            mouth_roi = cv2.resize(mouth_roi, (img_height, img_width))

            # Normalize the mouth ROI
            mouth_roi = mouth_roi / 255.
            mouth_roi_resized = cv2.resize(mouth_roi, (224, 224))
            prediction = modelyawn.predict(np.expand_dims(np.expand_dims(mouth_roi_resized, axis=0), axis=-1))


            #Classify yawning using the model
            #prediction = modelyawn.predict(np.expand_dims(np.expand_dims(mouth_roi, axis=0), axis=-1))


            # Draw a rectangle around the detected mouth and the predicted yawn label
            #color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)
            #cv2.rectangle(face_roi, (mx, my), (mx+mw, my+mh), color, 2)
            label = 'Yawning' if prediction > 0.8 else 'Not yawning'
            cv2.putText(frame, label, (10, img_height-50), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            return label
        # Draw a rectangle around the detected face
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    
        
while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    detect_eyes(frame,faces)
    yawndetect(frame,faces)
    x=detect_eyes(frame,faces)
    y=yawndetect(frame,faces)
    
    
    if x=='closed':
        cv2.putText(frame, 'Score'+str(Score), (100, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        Score += 1
        if Score > 15:
            try:
                   sound.play()
            except:
                    pass
        if y=='Yawning':
            if Score>5:
                try:
                    sound.play()
                except:
                    pass
    elif x=='open':
        cv2.putText(frame, 'Score'+str(Score), (100, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        Score -= 1
        if Score < 0:
            Score = 0
    
        
    
        
    

    # Show the frame with detections
    cv2.imshow('drowsiness detection', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:




