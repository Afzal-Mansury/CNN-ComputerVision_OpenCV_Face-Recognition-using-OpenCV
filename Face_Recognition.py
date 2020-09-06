# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 19:41:31 2020

@author: Admin
"""

# Transfer learning Example using VGG16

from keras.layers import Input, Lambda , Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# Re size all the images to below format
IMAGE_SIZE = [224,224]

train_path = "Face_Recognition/Train"
valid_path = "Face_Recognition/Test"

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights="imagenet",include_top=False)

# No need to train existing weights because is already trained
for layer in vgg.layers:
    layer.trainable = False
    
# usefull for getting number of classes
folders = glob("Face_Recognition/Train/*")    

# Our layers (we can add any number of layers)
x = Flatten()(vgg.output)
# prediction
prediction = Dense(len(folders),activation="softmax")(x)

#creating model
model = Model(inputs=vgg.input,outputs=prediction)

# Structure of model
model.summary()

# Model Compiling

model.compile(loss = "categorical_crossentropy",
                optimizer="adam",
                metrics = ["accuracy"])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip= True)

test_datagen =ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory("Face_Recognition/Train",
                                                 target_size=(224,224),
                                                 batch_size=32,
                                                 class_mode = "categorical")

test_set = train_datagen.flow_from_directory("Face_Recognition/Test",
                                                 target_size=(224,224),
                                                 batch_size=32,
                                                 class_mode = "categorical")

'''r=model.fit_generator(training_set, samples_per_epoch = 8000,
nb_epoch=5,validation_data = test_set, nb_val_samples = 2000)'''


# Fit the Model
r = model.fit_generator(training_set,
                        validation_data=test_set,
                        epochs=5,
                        steps_per_epoch= len(training_set),
                        validation_steps=len(test_set))

# loss

plt.plot(r.history["loss"],label = "train loss")
plt.plot(r.history["val_loss"],label = "val_loss")
plt.legend()
plt.show()
plt.savefig("LossVal_Loss")


# Accuracy
plt.plot(r.history["accuracy"],label = "train acc")
plt.plot(r.history["val_accuracy"],label = "val_acc")
plt.legend()
plt.show()
plt.savefig("AccVal_acc")

from keras.models import load_model
model.save("Face_Feacture_Model.h5")








