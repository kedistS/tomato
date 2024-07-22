from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../saved_models/model.keras", compile=False)
MODEL_L = tf.keras.models.load_model("../saved_models/leaf.keras", compile=False) 

CLASS_NAMES = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert('RGB')  # Ensure image is in RGB format
    image = image.resize((224, 224))  # Resize image for leaf detection model
    image = np.array(image) / 255.0  # Rescale image
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    # Predict if the image contains a leaf
    leaf_prediction = MODEL_L.predict(img_batch)
    if leaf_prediction[0] < 0.5: 
        raise HTTPException(status_code=400, detail="No leaf detected in the image")

    
    image = Image.open(BytesIO(await file.read())).convert('RGB')  # Ensure image is in RGB format
    image = image.resize((256, 256))  # Resize image for disease classification model
    image = np.array(image) / 255.0  # Rescale image
    img_batch = np.expand_dims(image, 0)

    # Predict the disease
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        'class': predicted_class,
        'confidence': confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)