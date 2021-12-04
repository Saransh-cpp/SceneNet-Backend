import cv2
from glob import glob
from PIL import Image
from keras.models import Sequential
from keras.applications.vgg19 import VGG19
from keras.layers import Input, Lambda, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator


folders = glob('indoorCVPR_09/Images/*')
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

# tell the model what cost and optimization method to use
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)


train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

training_set = train_datagen.flow_from_directory('indoorCVPR_09/Images/',
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 subset="training",
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('indoorCVPR_09/Images/',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            subset="validation",
                                            class_mode='categorical')

model.fit(
    training_set,
    validation_data=test_set,
    epochs=5,
    batch_size=32,
)

model.predict(cv2.imread(r"indoorCVPR_09\Images\airport_inside\airport_inside_0001.jpg"))
