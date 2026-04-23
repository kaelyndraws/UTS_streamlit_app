import requests

url = "http://127.0.0.1:8000/predict/all"
features = {
    "Gender_Male": 1,
    "Gender_Female": 0,
    "Branch_CSE": 1,
    "Branch_ECE": 0,
    "Branch_IT": 0,
    "Branch_ME": 0,
    "Branch_CE": 0,
    "cgpa": 8.0,
    "tenth_percentage": 74.0,
    "twelfth_percentage": 74.0,
    "backlogs": 0,
    "study_hours_per_day": 4.0,
    "attendance_percentage": 74.0,
    "projects_completed": 5,
    "internships_completed": 2,
    "coding_skill_rating": 3,
    "communication_skill_rating": 3,
    "aptitude_skill_rating": 4,
    "hackathons_participated": 3,
    "certifications_count": 2,
    "sleep_hours": 6.0,
    "stress_level": 6,
    "part_time_job": 0,
    "family_income_level": 1,
    "city_tier": 0,
    "internet_access": 1,
    "extracurricular_involvement": 1,
    "total_skills": 10,
    "academic_score": 76.0
}

response = requests.post(url, json = features)
print(response.status_code)
print(response.json())
