import os
import numpy as np
import splitfolders
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import load_model
from keras.applications.vgg19 import VGG19
from keras.layers import Input, Lambda, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator


# train test cross-validation split
if not os.path.exists(os.path.join(os.getcwd(), 'output')):
    splitfolders.ratio("indoorCVPR_09/Images", output="output",
                       seed=1337, ratio=(.7, .2, .1), group_prefix=None)

# calculate number of categories
folders = glob('indoorCVPR_09/Images/*')
categories = [folder.split("\\")[-1] for folder in folders]

# create a labels dict for ease
labels = {}
label_no = 0
for category in categories:
    labels[label_no] = category
    label_no += 1

# create a VGG19 net
vgg19 = VGG19(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
vgg19.trainable = False

# create extra layers for output
flatten_layer = Flatten()
prediction_layer = Dense(len(folders), activation='softmax')

# create the model
model = Sequential([
    vgg19,
    flatten_layer,
    prediction_layer
])

# view the structure of the model
model.summary()

# compile with adam
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

# create data generators for training and validation data
# testing has been done in the n`train.ipynb` notebook
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
)

# bring in the data
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

# fit the data
model.fit(
    training_set,
    validation_data=validation_set,
    epochs=10,
    batch_size=32,
)

# save the model
model.save('model.h5')
