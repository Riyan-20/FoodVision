import numpy as np
from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app=FastAPI()
static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Use TFSMLayer for Keras 3 compatibility with SavedModel
MODEL = tf.keras.layers.TFSMLayer(os.path.join(os.path.dirname(__file__), '..', 'model'), call_endpoint='serving_default')

CLASS_NAMES = [
    "apple_pie",
    "beef_tartare",
    "cannoli",
    "churros",
    "cup_cakes",
    "deviled_eggs",
    "donuts",
    "dumplings",
    "edamame",
    "french_fries",
    "fried_rice",
    "garlic_bread",
    "gnocchi",
    "greek_salad",
    "gyoza",
    "hamburger",
    "hot_and_sour_soup",
    "hot_dog",
    "ice_cream",
    "mussels",
    "onion_rings",
    "oysters",
    "pancakes",
    "pizza",
    "ramen",
    "samosa",
    "sashimi",
    "spring_rolls",
    "sushi",
    "waffles"
]

def preprocess_image(image: np.ndarray):
    # If image is a path, load it, else assume it's already an array
    img = Image.fromarray(image)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, 0)  # Create a batch
    return img_array.astype(np.float32)

@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/ping")
async def ping():
    return "Hello, this is Ryan"

def read_file_as_image(data)->np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
        file:UploadFile=File(...)
):  
    image=read_file_as_image(await file.read())
    preprocessed_image = preprocess_image(image)
    # Use TFSMLayer for inference
    predictions = MODEL(tf.convert_to_tensor(preprocessed_image))
    # Output is a dict with key 'dense_2'
    preds = predictions['dense_2']
    predicted_class_index = int(tf.argmax(preds[0]).numpy())
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    return{
        'class': predicted_class_name
    }

if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8088)

