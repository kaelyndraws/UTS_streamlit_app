import streamlit as st
import joblib
import numpy as np
import pandas as pd

scaler = joblib.load("artifacts/preprocessor.pkl")
model_class = joblib.load("models/classification_model.pkl")
model_reg = joblib.load("models/regression_model.pkl")
# mlb = joblib.load("mlb.pkl")

def make_prediction_class(features_class):
    input_array_class = pd.DataFrame([features_class])
    # input_array_class = pd.get_dummies(input_array_class)
    cols = joblib.load("artifacts/columns_class.pkl")
    input_array_class = input_array_class.reindex(columns = cols, fill_value = 0)
    x_scaled_class = scaler.transform(input_array_class)
    prediction_class = model_class.predict(x_scaled_class)

    return prediction_class[0]

def make_prediction_reg(features_reg):
    input_array_reg = pd.DataFrame([features_reg])
    # input_array_reg = pd.get_dummies(input_array_reg)
    cols = joblib.load("artifacts/columns_reg.pkl")
    input_array_reg = input_array_reg.reindex(columns = cols, fill_value = 0)
    x_scaled_reg = scaler.transform(input_array_reg)
    prediction_reg = model_reg.predict(x_scaled_reg)

    return prediction_reg[0]

def main():
    st.title("Machine Learning Placement Status and Salary (LPA) Prediction Model Deployment")

    gender = st.selectbox(
        "Gender", [
            "Male",
            "Female"
        ]
    )
    genders = ["Male", "Female"]
    gender_features = {f"Gender_{g}":0 for g in genders}
    gender_features[f"Gender_{gender}"] = 1
    branch = st.selectbox(
        "Branch", [
            "CSE",
            "ECE",
            "IT",
            "ME",
            "CE"
        ]
    )
    branches = ["CSE", "ECE", "IT", "ME", "CE"]
    branch_features = {f"Branch_{b}":0 for b in branches}
    branch_features[f"Branch_{branch}"] = 1
    cgpa = st.number_input("CGPA", min_value = 0.0, max_value = 10.0, value = 8.0)
    tenth_percentage = st.number_input("Tenth Grade Score", min_value = 0.0, max_value = 100.0, value = 74.0)
    twelfth_percentage = st.number_input("Twelfth Grade Score", min_value = 0.0, max_value = 100.0, value = 74.0)
    backlogs = st.number_input("Failed Courses", min_value = 0, max_value = 5, value = 0)
    study_hours_per_day = st.number_input("Number of Study Hours per Day", min_value = 0.0, max_value = 10.0, value = 4.0)
    attendance_percentage = st.number_input("Attendance Percentage", min_value = 0.0, max_value = 100.0, value = 74.0)
    projects_completed = st.number_input("Number of Projects Completed", min_value = 0, max_value = 8, value = 5)
    internships_completed = st.number_input("Number of Internships Completed", min_value = 0, max_value = 4, value = 2)
    coding_skill_rating = st.number_input("Coding Skill Rating", min_value = 1, max_value = 5, value = 3)
    communication_skill_rating = st.number_input("Communication Skill Rating", min_value = 1, max_value = 5, value = 3)
    aptitude_skill_rating = st.number_input("Aptitude Skill Rating", min_value = 1, max_value = 5, value = 4)
    hackathons_participated = st.number_input("Number of Hackathons Participated", min_value = 0, max_value = 6, value = 3)
    certifications_count = st.number_input("Certifications Count", min_value = 0, max_value = 9, value = 2)
    sleep_hours = st.number_input("Number of Sleep Hours", min_value = 4.0, max_value = 9.0, value = 6.0)
    stress_level = st.number_input("Stress Level", min_value = 1, max_value = 10, value = 6)
    part_time_job = st.selectbox(
        "Part Time Job", [
            "No",
            "Yes"
        ]
    )
    part_time_job_encoded = {
        "No": 0,
        "Yes": 1
    }[part_time_job]
    family_income_level = st.selectbox(
        "Family Income Level", [
            "Low",
            "Medium",
            "High"
        ]
    )
    family_income_level_encoded = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }[family_income_level]
    city_tier = st.selectbox(
        "City Tier", [
            "Tier 1",
            "Tier 2",
            "Tier 3"
        ]
    )
    city_tier_encoded = {
        "Tier 1": 0,
        "Tier 2": 1,
        "Tier 3": 2
    }[city_tier]
    internet_access = st.selectbox(
        "Internet Access", [
            "No",
            "Yes"
        ]
    )
    internet_access_encoded = {
        "No": 0,
        "Yes": 1
    }[internet_access]
    extracurricular_involvement = st.selectbox(
        "Extracurricular Involvement", [
            "Low",
            "Medium",
            "High"
        ]
    )
    extracurricular_involvement_encoded = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }[extracurricular_involvement]
    total_skills = coding_skill_rating + communication_skill_rating + aptitude_skill_rating
    academic_score = (tenth_percentage + twelfth_percentage + (cgpa * 10)) / 3
    
    if st.button("Make Prediction"):
        features = {
            **gender_features,
            **branch_features,
            "cgpa": cgpa,
            "tenth_percentage": tenth_percentage,
            "twelfth_percentage": twelfth_percentage,
            "backlogs": backlogs,
            "study_hours_per_day": study_hours_per_day,
            "attendance_percentage": attendance_percentage,
            "projects_completed": projects_completed,
            "internships_completed": internships_completed,
            "coding_skill_rating": coding_skill_rating,
            "communication_skill_rating": communication_skill_rating,
            "aptitude_skill_rating": aptitude_skill_rating,
            "hackathons_participated": hackathons_participated,
            "certifications_count": certifications_count,
            "sleep_hours": sleep_hours,
            "stress_level": stress_level,
            "part_time_job": part_time_job_encoded,
            "family_income_level": family_income_level_encoded,
            "city_tier": city_tier_encoded,
            "internet_access": internet_access_encoded,
            "extracurricular_involvement": extracurricular_involvement_encoded,
            "total_skills": total_skills,
            "academic_score": academic_score
        }
        result_class = make_prediction_class(features)
        result_reg = make_prediction_reg(features)

        st.success(f"The prediction for placement status is: {result_class}\nThe prediction for salary (LPA) is: {result_reg}")

if __name__ == "__main__":
    main()
    
