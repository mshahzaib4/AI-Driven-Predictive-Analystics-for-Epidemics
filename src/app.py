from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import pickle
import os
import joblib
import pandas as pd

# Get the absolute path to the parent of /src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize FastAPI and Jinja2 template directory
app = FastAPI()

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/documentation", response_class=HTMLResponse)
async def documentation(request: Request):
    return templates.TemplateResponse("documentation.html", {"request": request})

@app.get("/profile", response_class=HTMLResponse)
async def profile(request: Request):
    return templates.TemplateResponse("profile.html", {"request": request})

# ==================== LUNG CANCER ====================
# Load model and scaler from Models directory
lung_model_path = os.path.join("..", "Models", "lung_cancer_model.pkl")
lung_scaler_path = os.path.join("..", "Models", "age_scaler.pkl")

with open(lung_model_path, "rb") as f:
    lung_model = pickle.load(f)

with open(lung_scaler_path, "rb") as f:
    lung_scaler = pickle.load(f)

# Convert Yes/No to integer
def yes_no_to_int(value: str) -> int:
    return 1 if value.strip().lower() == "yes" else 0

# Convert gender to integer
def gender_to_int(value: str) -> int:
    return 1 if value.strip().lower() == "male" else 0

# GET route to show the form
@app.get("/lung_cancer", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("lung cancer.html", {"request": request, "prediction": None})

# POST route to handle prediction - KEPT ORIGINAL PATH /predict
@app.post("/predict", response_class=HTMLResponse)
def predict_lung(request: Request,
                NAME: str = Form(...),
                GENDER: str = Form(...),
                AGE: int = Form(...),
                SMOKING: str = Form(...),
                YELLOW_FINGERS: str = Form(...),
                ANXIETY: str = Form(...),
                PEER_PRESSURE: str = Form(...),
                CHRONIC_DISEASE: str = Form(...),
                FATIGUE: str = Form(...),
                ALLERGY: str = Form(...),
                WHEEZING: str = Form(...),
                ALCOHOL_CONSUMING: str = Form(...),
                COUGHING: str = Form(...),
                SHORTNESS_OF_BREATH: str = Form(...),
                SWALLOWING_DIFFICULTY: str = Form(...),
                CHEST_PAIN: str = Form(...)):

    # Convert inputs to numerical values
    gender = gender_to_int(GENDER)
    smoking = yes_no_to_int(SMOKING)
    yellow_fingers = yes_no_to_int(YELLOW_FINGERS)
    anxiety = yes_no_to_int(ANXIETY)
    peer_pressure = yes_no_to_int(PEER_PRESSURE)
    chronic_disease = yes_no_to_int(CHRONIC_DISEASE)
    fatigue = yes_no_to_int(FATIGUE)
    allergy = yes_no_to_int(ALLERGY)
    wheezing = yes_no_to_int(WHEEZING)
    alcohol = yes_no_to_int(ALCOHOL_CONSUMING)
    coughing = yes_no_to_int(COUGHING)
    short_breath = yes_no_to_int(SHORTNESS_OF_BREATH)
    swallowing = yes_no_to_int(SWALLOWING_DIFFICULTY)
    chest_pain = yes_no_to_int(CHEST_PAIN)

    # Scale age
    scaled_age = lung_scaler.transform([[AGE]])[0][0]

    # Create feature vector
    features = np.array([[gender, scaled_age, smoking, yellow_fingers, anxiety, peer_pressure,
                        chronic_disease, fatigue, allergy, wheezing, alcohol,
                        coughing, short_breath, swallowing, chest_pain]])
   
    # Predict and calculate probability
    prediction = lung_model.predict(features)[0]
    
    probability = lung_model.predict_proba(features)[0][1] * 100  # Probability of positive class

    # Render the result
    return templates.TemplateResponse("lung cancer.html", {
        "request": request,
        "prediction": prediction,
        "probability": round(probability, 2),
        "name": NAME
    })

# ==================== HEART DISEASE ====================
# Load model and columns once at startup
heart_model_path = os.path.join("..", "Models", "heart_disease_model.pkl")
heart_columns_path = os.path.join("..", "Models", "heart_model_columns.pkl")

heart_model = joblib.load(heart_model_path)
heart_model_columns = joblib.load(heart_columns_path)

@app.get("/heart_disease", response_class=HTMLResponse)
def heart_form_get(request: Request):
    return templates.TemplateResponse("heart_disease.html", {"request": request})

# POST route - KEPT ORIGINAL PATH /predict_heart
@app.get("/test-positive-heart", response_class=HTMLResponse)
def test_positive_heart(request: Request):
    # Example input data that likely results in a positive heart disease prediction
    input_data = {
        'Age': 65,
        'Gender': 'male',
        'Blood Pressure': 150,
        'Cholesterol Level': 250,
        'Exercise Habits': 'none',
        'Smoking': 'yes',
        'Family Heart Disease': 'yes',
        'Diabetes': 'yes',
        'BMI': 30.5,
        'High Blood Pressure': 'yes',
        'High LDL Cholesterol': 'yes',
        'Alcohol Consumption': 'yes',
        'Stress Level': 'high',
        'Sleep Hours': 5,
        'Sugar Consumption': 'high',
        'Triglyceride Level': 200,
        'Fasting Blood Sugar': 130,
        'CRP Level': 3.0,
        'Homocysteine Level': 15.0
    }

    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=heart_model_columns, fill_value=0)

    prediction = heart_model.predict(df)[0]
    probability = heart_model.predict_proba(df)[0][1] * 100

    return templates.TemplateResponse("heart_disease.html", {
        "request": request,
        "prediction": prediction,
        "probability": round(probability, 2),
        "name": "Test Positive"
    })


@app.get("/test-negative-heart", response_class=HTMLResponse)
def test_negative_heart(request: Request):
    # Example input data that likely results in a negative heart disease prediction
    input_data = {
        'Age': 30,
        'Gender': 'female',
        'Blood Pressure': 110,
        'Cholesterol Level': 180,
        'Exercise Habits': 'regular',
        'Smoking': 'no',
        'Family Heart Disease': 'no',
        'Diabetes': 'no',
        'BMI': 22.0,
        'High Blood Pressure': 'no',
        'High LDL Cholesterol': 'no',
        'Alcohol Consumption': 'no',
        'Stress Level': 'low',
        'Sleep Hours': 8,
        'Sugar Consumption': 'low',
        'Triglyceride Level': 100,
        'Fasting Blood Sugar': 90,
        'CRP Level': 0.5,
        'Homocysteine Level': 5.0
    }

    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=heart_model_columns, fill_value=0)

    prediction = heart_model.predict(df)[0]
    probability = heart_model.predict_proba(df)[0][1] * 100

    return templates.TemplateResponse("heart_disease.html", {
        "request": request,
        "prediction": prediction,
        "probability": round(probability, 2),
        "name": "Test Negative"
    })

# ==================== LIVER DISEASE ====================
# Load liver model once at startup

model_path = os.path.join("..", "Models", "liver_disease_model.pkl")
with open(model_path, "rb") as f:
    liver_model = pickle.load(f)

@app.get("/liver_disease", response_class=HTMLResponse)
async def liver_form(request: Request):
    return templates.TemplateResponse("liver_disease.html", {
        "request": request,
        "prediction": None,
        "probability": None,
        "name": ""
    })

@app.post("/predict_liver", response_class=HTMLResponse)
async def predict_liver(
    request: Request,
    Name: str = Form(...),
    Age: float = Form(...),
    Gender: str = Form(...),
    Total_Bilirubin: float = Form(...),
    Direct_Bilirubin: float = Form(...),
    Alkaline_Phosphotase: float = Form(...),
    Alamine_Aminotransferase: float = Form(...),
    Aspartate_Aminotransferase: float = Form(...),
    Total_Protiens: float = Form(...),
    Albumin: float = Form(...),
    Albumin_and_Globulin_Ratio: float = Form(...)
):
    # Convert gender to numeric
    gender_val = 1 if Gender.strip().lower() == "male" else 0

    input_data = np.array([[Age, gender_val, Total_Bilirubin, Direct_Bilirubin,
                            Alkaline_Phosphotase, Alamine_Aminotransferase,
                            Aspartate_Aminotransferase, Total_Protiens,
                            Albumin, Albumin_and_Globulin_Ratio]])

    prediction = int(liver_model.predict(input_data)[0])
    probability = round(liver_model.predict_proba(input_data)[0][1] * 100, 2)

    return templates.TemplateResponse("liver_disease.html", {
        "request": request,
        "name": Name,
        "prediction": prediction,
        "probability": probability
    })
