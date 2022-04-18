# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 12:48:18 2022

@author: ppallapotu
"""
import cv2
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob#for checking how many folder are present in the respecctive current folder

import matplotlib.pyplot as plt

vgg=VGG16(include_top=False,weights='imagenet',input_shape=(224,224,3),classes=6)
for layers in vgg.layers:
    layers.trainable=False
x=Flatten()(vgg.output)
Preduction=Dense(6,activation='softmax',)(x)
model=Model(inputs=vgg.input,outputs=Preduction)
model.summary()
model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')
model.summary()
file=glob("D:\\fruit\\*")
traindata=ImageDataGenerator(shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
testdata=ImageDataGenerator(rescale=1./255)
train_set=traindata.flow_from_directory('D:\\fruit\\train',target_size=(224,224),batch_size=32,class_mode='categorical')
test_set=testdata.flow_from_directory('D:\\fruit\\test',target_size=(224,224),batch_size=32,class_mode='categorical')
r = model.fit(
  train_set,
  validation_data=test_set,
  epochs=2,
  
)
from keras.callbacks import ModelCheckpoint,EarlyStopping
filename="saved_model/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
ModelCheckpoint(filename,monitor='val_loss',verbode=1,save_best_only=True ,mode='max')
# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
model.save("fruit.h5")#for saving the model

 
        
    










