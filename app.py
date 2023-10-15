from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from generationml import prediction
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
import numpy as np
from keras.models import load_model
import json

app = FastAPI()



model = load_model('plant_disease_model.h5')


# Specify the path to the JSON file you want to read
json_file_path = "class_indices.json"  # Update with your file path

# Read the JSON data from the file
with open(json_file_path, "r") as json_file:
    ref = json.load(json_file)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Save the uploaded image to a temporary file
    with open(file.filename, "wb") as image_file:
        image_file.write(file.file.read())

    # Call the prediction function with the image path
    # result = model.predict(file.filename)
    result = prediction(file.filename)

    return JSONResponse(content={"prediction": result})


def prediction(path):
  img = load_img(path, target_size=(256,256))
  i = img_to_array(img)
  im = preprocess_input(i)
  img = np.expand_dims(im, axis=0)
  pred = np.argmax(model.predict(img))
  return (f'Bu bitkinin xəstə olub olmaması: {ref[str(pred)]}')

@app.get("/")
def get_message():
    return {"message": "Hello, FastAPI!"}