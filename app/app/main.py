from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent

# Load model
model_path = BASE_DIR / "model.joblib"
model = joblib.load(model_path)

app = FastAPI(title="Insurance Charges Predictor")

# Static files and templates
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class InsuranceFeatures(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(features: InsuranceFeatures):
    data = pd.DataFrame([{
        "age": features.age,
        "sex": features.sex,
        "bmi": features.bmi,
        "children": features.children,
        "smoker": features.smoker,
        "region": features.region,
    }])

    pred = model.predict(data)[0]
    return {"predicted_charge": float(pred)}
