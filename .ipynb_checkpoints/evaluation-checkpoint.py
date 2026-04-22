import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR

def evaluate(test_scaled_class, test_scaled_reg, run_id):
    x_test_class = test_scaled_class.drop("placement_status", axis = 1)
    y_test_class = test_scaled_class["placement_status"]
    x_test_reg = test_scaled_reg.drop("salary_lpa", axis = 1)
    y_test_reg = test_scaled_reg["salary_lpa"]
    model_class = mlflow.sklearn.load_model(f"runs:/{run_id}/classification_model")
    model_reg = mlflow.sklearn.load_model(f"runs:/{run_id}/regression_model")
    predictions_class = model_class.predict(x_test_class)
    predictions_class = predictions_class.flatten()
    predictions_reg = model_reg.predict(x_test_reg)
    predictions_reg = predictions_reg.flatten()
    y_test_class = y_test_class.values
    y_test_reg = y_test_reg.values
    accuracy = accuracy_score(y_test_class, predictions_class)
    precision = precision_score(y_test_class, predictions_class, average = "weighted")
    recall = recall_score(y_test_class, predictions_class, average = "weighted")
    f1 = f1_score(y_test_class, predictions_class, average = "weighted")
    mae = mean_absolute_error(y_test_reg, predictions_reg)
    mse = mean_squared_error(y_test_reg, predictions_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, predictions_reg))
    r2 = r2_score(y_test_reg, predictions_reg)

    with mlflow.start_run(run_id = run_id):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("mean_absolute_error", mae)
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("root_mean_squared_error", rmse)
        mlflow.log_metric("r2_score", r2)

    print(f"Evaluation completed | Accuracy = {accuracy:.3f} | Mean Absolute Error = {mae:.3f}")

    return accuracy, precision, recall, f1, mae, mse, rmse, r2

if __name__ == "__main__":
    evaluate()