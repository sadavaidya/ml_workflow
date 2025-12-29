from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.inference import load_model, predict


app = FastAPI(
    title="Insurance Cost Prediction API",
    description="Production-style ML inference API",
    version="0.1.0",
)


# Load model once at startup
model = load_model()


class InsuranceInput(BaseModel):
    age: int
    bmi: float
    children: int
    sex_male: int
    smoker_yes: int
    region_northwest: int
    region_southeast: int
    region_southwest: int


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_cost(data: InsuranceInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = predict(model, input_df)

    return {
        "predicted_cost": float(prediction[0])
    }
