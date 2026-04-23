from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title = "Placement and Salary Prediction API")
scaler = joblib.load("artifacts/preprocessor.pkl")
model_class = joblib.load("models/classification_model.pkl")
model_reg = joblib.load("models/regression_model.pkl")
cols_class = joblib.load("artifacts/columns_class.pkl")
cols_reg = joblib.load("artifacts/columns_reg.pkl")

class PredictionRequest(BaseModel):
  Gender_Male: float
  Gender_Female: float
  Branch_CSE: float
  Branch_ECE: float
  Branch_IT: float
  Branch_ME: float
  Branch_CE: float
  cgpa: float
  tenth_percentage: float
  twelfth_percentage: float
  backlogs: int
  study_hours_per_day: float
  attendance_percentage: float
  projects_completed: int
  internships_completed: int
  coding_skill_rating: int
  communication_skill_rating: int
  aptitude_skill_rating: int
  hackathons_participated: int
  certifications_count: int
  sleep_hours: float
  stress_level: int
  part_time_job: int
  family_income_level: int
  city_tier: int
  internet_access: int
  extracurricular_involvement: int
  total_skills: int
  academic_score: float

def preprocess_for_classification(data: PredictionRequest):
  input_df_class = pd.DataFrame([data.model_dump()])
  input_df_class = input_df_class.reindex(columns = cols_class, fill_value = 0)
  x_scaled_class = scaler.transform(input_df_class)

  return x_scaled_class

def preprocess_for_regression(data: PredictionRequest):
  input_df_reg = pd.DataFrame([data.model_dump()])
  input_df_reg = input_df_reg.reindex(columns = cols_reg, fill_value = 0)
  x_scaled_reg = scaler.transform(input_df_reg)

  return x_scaled_reg

@app.get("/")

def root():
  return {"message": "Placement and Salary Prediction API is running."}

@app.post("/predict/all")
def predict(data: PredictionRequest):
  x_scaled_class = preprocess_for_classification(data)
  x_scaled_reg = preprocess_for_regression(data)
  pred_class = model_class.predict(x_scaled_class)[0]
  pred_reg = model_reg.predict(x_scaled_reg)[0]
  result = {
    "classification": {
      "prediction": int(pred_class),
      "label": "Placed" if int(pred_class) == 1 else "Not Placed"
    },
    "regression": {
      "prediction": float(pred_reg),
      "unit": "LPA"
    }
  }

  if hasattr(model_class, "predict_proba"):
    proba = model_class.predict_proba(x_scaled_class)[0]
    result["classification"]["probability_not_placed"] = float(proba[0])
    result["classification"]["probability_placed"] = float(proba[1])

  return result
