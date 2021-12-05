import cv2
import numpy as np
from keras.models import load_model
from fastapi import FastAPI, File, UploadFile


app = FastAPI()

labels = {
    0: "airport_inside",
    1: "artstudio",
    2: "auditorium",
    3: "bakery",
    4: "bar",
    5: "bathroom",
    6: "bedroom",
    7: "bookstore",
    8: "bowling",
    9: "buffet",
    10: "casino",
    11: "children_room",
    12: "church_inside",
    13: "classroom",
    14: "cloister",
    15: "closet",
    16: "clothingstore",
    17: "computerroom",
    18: "concert_hall",
    19: "corridor",
    20: "deli",
    21: "dentaloffice",
    22: "dining_room",
    23: "elevator",
    24: "fastfood_restaurant",
    25: "florist",
    26: "gameroom",
    27: "garage",
    28: "greenhouse",
    29: "grocerystore",
    30: "gym",
    31: "hairsalon",
    32: "hospitalroom",
    33: "inside_bus",
    34: "inside_subway",
    35: "jewelleryshop",
    36: "kindergarden",
    37: "kitchen",
    38: "laboratorywet",
    39: "laundromat",
    40: "library",
    41: "livingroom",
    42: "lobby",
    43: "locker_room",
    44: "mall",
    45: "meeting_room",
    46: "movietheater",
    47: "museum",
    48: "nursery",
    49: "office",
    50: "operating_room",
    51: "pantry",
    52: "poolinside",
    53: "prisoncell",
    54: "restaurant",
    55: "restaurant_kitchen",
    56: "shoeshop",
    57: "stairscase",
    58: "studiomusic",
    59: "subway",
    60: "toystore",
    61: "trainstation",
    62: "tv_studio",
    63: "videostore",
    64: "waitingroom",
    65: "warehouse",
    66: "winecellar",
}


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
