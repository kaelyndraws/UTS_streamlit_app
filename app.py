import streamlit as st
import joblib
import numpy as np
import pandas as pd

scaler = joblib.load("artifacts/preprocessor.pkl")
model_class = joblib.load("artifacts/model_class.pkl")
model_reg = joblib.load("artifacts/model_reg.pkl")
mlb = joblib.load("mlb.pkl")

def make_prediction_class(features_class):
    input_array_class = np.array(features_class).reshape(1, -1)
    x_scaled_class = scaler.transform(input_array_class)
    prediction_class = model_class.predict(x_scaled_class)

    return prediction_class[0]

def make_prediction_reg(features_reg):
    input_array_reg = np.array(features_reg).reshape(1, -1)
    x_scaled_reg = scaler.transform(input_array_reg)
    prediction_reg = model_reg.predict(x_scaled_reg)

    return prediction_reg[0]

def main():
    st.title("Machine Learning Placement Status and Salary (LPA) Prediction Model Deployment")

    gender = st.multiselect(
        "Gender", [
            "Male",
            "Female"
        ]
    )
    genders = ["Male", "Female"]
    gender_features = {f"Gender_{g}":0 for g in genders}
    branch = st.multiselect(
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
    part_time_job = st.multiselect(
        "Part Time Job", [
            "No",
            "Yes"
        ]
    )
    part_time_job_encoded = {
        "No": 0,
        "Yes": 1
    }[part_time_job]
    family_income_level = st.multiselect(
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
    city_tier = st.multiselect(
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
    internet_access = st.multiselect(
        "Internet Access", [
            "No",
            "Yes"
        ]
    )
    internet_access_encoded = {
        "No": 0,
        "Yes": 1
    }[internet_access]
    extracurricular_involvement = st.multiselect(
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
        features = [gender_features, branch_features, cgpa, tenth_percentage, backlogs, study_hours_per_day, attendance_percentage, projects_completed, internships_completed, coding_skill_rating, communication_skill_rating, aptitude_skill_rating, hackathons_participated, certifications_count, sleep_hours, stress_level, part_time_job_encoded, family_income_level_encoded, city_tier_encoded, internet_access_encoded, extracurricular_involvement_encoded, total_skills, academic_score]
        result_class = make_prediction_class(features)
        result_reg = make_prediction_reg(features)

        st.success(f"The prediction for placement status is: {result_class}\nThe prediction for salary (LPA) is: {result_reg}")

if __name__ == "__main__":
    main()
    
