## SceneNet Backend

Scenery detection using transfer learning.

## Description
The API uses the `VGG19` convoution neural network, which is trained on a dataset of 10903 images belonging to 67 different classes.
The classes (as used in the code) -
```py
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
```

## Usage
- The API can be accessed through the URL - https://scene-net.herokuapp.com/
- To predict an image's class, use the `/predict` endpoint
- For the complete documentation refer to - https://scene-net.herokuapp.com/docs

## Running locally
### To train the model locally -
1. Fork and clone the repository
```
git clone https://github.com/<your_username>/SceneNet-Backend
```
2. Create a new virtual environment
```
python -m venve .venv
```
3. Activate the virtual environment
```
.venv/Scripts/activate
```
4. Install requirements for training
```
python -m pip install -r train_model/train_requirements.txt
```
5. Run the jupyter in the virtual environment
```
ipython kernel install --user --name=venv
# select the kernel named after your virtual environment in jupyter notebook
```
### To run the API locally-
1. Fork and clone the repository
```
git clone https://github.com/<your_username>/SceneNet-Backend
```
2. Create a new virtual environment
```
python -m venve .venv
```
3. Activate the virtual environment
```
.venv/Scripts/activate
```
4. Install requirements for training (the `Heroku` deployment uses `tensorflow-cpu` and `opencv-python-headless` because of the memory limitations, but you can switch to `tensorflow` and `opencv-python` if you are running this locally)
```
python -m pip install -r requirements.txt
```
5. Fire up the API
```
uvicorn backend.backend:app --reload
```

## Dataset used

https://www.kaggle.com/itsahmad/indoor-scenes-cvpr-2019
