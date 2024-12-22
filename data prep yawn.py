#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import shutil

# Path to the folder containing the images
src_folder = 'C://Users//lalga//Downloads//yawndetectdata//no_yawning'
# Path to the folder where you want to move the images
dest_folder = 'C://Users//lalga//Downloads//yawndetectdata//train//yawning'

# Get a list of all the image files in the source folder
image_files = [os.path.join(src_folder, f) for f in os.listdir(src_folder) if f.endswith('.jpg')]

# Calculate the number of images to move
num_images = len(image_files)
print(num_images)


# 2528 - yawning
# 2591 - no yawning

# In[5]:


import os
import shutil

# Path to the folder containing the images
src_folder = 'C://Users//lalga//Downloads//yawndetectdata//yawning'
# Path to the folder where you want to move the images
dest_folder = 'C://Users//lalga//Downloads//yawndetectdata//test//yawning'

# Get a list of all the image files in the source folder
image_files = [os.path.join(src_folder, f) for f in os.listdir(src_folder) if f.endswith('.jpg')]





# Loop through the images to move and move them to the destination folder
for image_file in image_files[1896:]:
    shutil.copy(image_file, dest_folder)


# In[7]:


import os
import shutil

# Path to the folder containing the images
src_folder = 'C://Users//lalga//Downloads//yawndetectdata//no_yawning'
# Path to the folder where you want to move the images
dest_folder = 'C://Users//lalga//Downloads//yawndetectdata//test//no_yawning'

# Get a list of all the image files in the source folder
image_files = [os.path.join(src_folder, f) for f in os.listdir(src_folder) if f.endswith('.jpg')]



# Loop through the images to move and move them to the destination folder
for image_file in image_files[1943:]:
    shutil.copy(image_file, dest_folder)


# In[1]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report


# In[2]:


dataset_dir = 'C://Users//lalga//Downloads//yawndetectdata//'


# In[3]:


# Set the batch size and image dimensions
batch_size = 32
img_height, img_width = 299, 299


# In[4]:


# Define the number of classes (yawn or no-yawn)
num_classes = 2


# In[5]:


# Set up data augmentation and normalization for training and validation data
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, horizontal_flip=True, vertical_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[6]:


# Create the data generators for training and validation data
train_generator = train_datagen.flow_from_directory(os.path.join(dataset_dir, 'train'), target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary')
test_generator = test_datagen.flow_from_directory(os.path.join(dataset_dir, 'test'), target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary')


# In[7]:


# Load the Inception-V3 model with pre-trained weights from ImageNet, excluding the top classification layer
base_model = InceptionV3(weights='imagenet', include_top=False)


# In[8]:


# Add a global average pooling layer and a dense layer with sigmoid activation for binary classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1, activation='sigmoid')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=x)


# In[9]:


# Freeze the pre-trained layers to only train the new top layers
for layer in base_model.layers:
    layer.trainable = False


# In[10]:


# Compile the model with binary cross-entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

# Train the model on the training data and validate on the validation data
history = model.fit(train_generator, epochs=10, validation_data=test_generator)


# model 2
