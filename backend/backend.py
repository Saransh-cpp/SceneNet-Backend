import cv2
import numpy as np
from labels import labels
from keras.models import load_model
from fastapi import FastAPI, File, UploadFile


app = FastAPI()


@app.get("/")
def home():
    return {"message": "Please refer to the README for more information."}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    with open("image.jpg", "wb+") as f:
        f.write(image.file.read())

    img = cv2.imread("image.jpg")
    img = cv2.resize(img, (224, 224))

    img = img / 255.0

    model = load_model("../train_mode/model.h5")
    y_pred = model.predict(np.array([img]))

    category = labels[np.argmax(y_pred.flatten())]

    return {"category": category}
