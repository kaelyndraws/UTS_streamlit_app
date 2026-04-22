import pandas as pd
import mlflow
import mlflow.sklearn
from catboost import CatBoostClassifier, CatBoostRegressor
import joblib
import os
import numpy as np

def train(train_scaled_class, train_scaled_reg):
    mlflow.set_experiment("Streamlit-Pipeline")

    x_train_class = train_scaled_class.drop("placement_status", axis = 1)
    y_train_class = train_scaled_class["placement_status"]
    x_train_reg = train_scaled_reg.drop("salary_lpa", axis = 1)
    y_train_reg = train_scaled_reg["salary_lpa"]

    with mlflow.start_run() as run:
        model_class = CatBoostClassifier(
            learning_rate = 0.05,
            l2_leaf_reg = 3,
            iterations = 100,
            depth = 6,
            verbose = 0
        )
        model_reg = CatBoostRegressor(
            verbose = 0,
            learning_rate = 0.05,
            l2_leaf_reg = 5,
            iterations = 200,
            depth = 4,
            bagging_temperature = 1
        )
        
        model_class.fit(x_train_class, y_train_class)
        model_reg.fit(x_train_reg, y_train_reg)
        mlflow.log_param("class_learning_rate", 0.05)
        mlflow.log_param("class_l2_leaf_reg", 3)
        mlflow.log_param("class_iterations", 100)
        mlflow.log_param("class_depth", 6)
        mlflow.log_param("class_verbose", 0)
        mlflow.log_param("reg_learning_rate", 0.05)
        mlflow.log_param("reg_l2_leaf_reg", 5)
        mlflow.log_param("reg_iterations", 200)
        mlflow.log_param("reg_depth", 4)
        mlflow.log_param("reg_verbose", 0)
        mlflow.log_param("reg_bagging_temperature", 1)
        mlflow.sklearn.log_model(sk_model = model_class, artifact_path = "classification_model")
        mlflow.sklearn.log_model(sk_model = model_reg, artifact_path = "regression_model")
        os.makedirs("models", exist_ok = True)
        joblib.dump(model_class, "models/classification_model.pkl", compress = 3)
        joblib.dump(model_reg, "models/regression_model.pkl", compress = 3)
        
        return run.info.run_id
    
if __name__ == "__main__":
    train(train_scaled_class, train_scaled_reg)