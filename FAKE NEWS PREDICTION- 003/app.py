from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np

# Load the saved model using joblib
model = load('best_model.joblib')

# Define FastAPI app 
app = FastAPI()

# Define input data model
class Item(BaseModel):
    text: str

# Endpoint for prediction
@app.post("/predict")
async def predict(item: Item):
    text = item.text
    prediction = model.predict([text])[0]  #  model.predict returns a single prediction
    return {"prediction": int(prediction)}  # Ensure prediction is converted to native Python int
