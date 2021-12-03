from fastapi import FastAPI, File, UploadFile


app = FastAPI()


@app.get("/")
def home():
    return {"message": "Please refer to the README for more information."}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    with open("image.jpg", "wb+") as f:
        f.write(image.file.read())

    return {
        "text": "working"
    }
