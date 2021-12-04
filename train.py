import os
import pickle
import numpy as np
import splitfolders
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.applications.vgg19 import VGG19
from keras.layers import Input, Lambda, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator


if not os.path.exists(os.path.join(os.getcwd(), 'output')):
    splitfolders.ratio("indoorCVPR_09/Images", output="output",
                       seed=1337, ratio=(.7, .2, .1), group_prefix=None)

folders = glob('indoorCVPR_09/Images/*')
categories = [folder.split("\\")[-1] for folder in folders]

labels = {}
label_no = 0
for category in categories:
    labels[label_no] = category
    label_no += 1

vgg19 = VGG19(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
vgg19.trainable = False

flatten_layer = Flatten()
prediction_layer = Dense(len(folders), activation='softmax')

model = Sequential([
    vgg19,
    flatten_layer,
    prediction_layer
])

# view the structure of the model
model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
)

training_set = train_datagen.flow_from_directory(
    'output/train/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_set = val_datagen.flow_from_directory(
    'output/val/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

model.fit(
    training_set,
    validation_data=validation_set,
    epochs=10,
    batch_size=32,
)

from keras.models import load_model

model.save('model.h5')
