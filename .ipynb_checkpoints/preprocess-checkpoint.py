import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess():
    os.makedirs("artifacts", exist_ok=True)

    df = pd.read_csv("ingested/df_cleaned.csv")
    x = df.drop(["placement_status", "salary_lpa"], axis=1)
    y_class = df["placement_status"]
    y_reg = df["salary_lpa"]
    x_train_class, x_test_class, y_train_class, y_test_class = train_test_split(
        x, y_class, test_size = 0.2, random_state = 42, 
        stratify = y_class
    )
    x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
        x, y_reg, test_size = 0.2, random_state = 42
    )
    scaler = StandardScaler()
    scaler_train_class = pd.DataFrame(
        scaler.fit_transform(x_train_class),
        columns = x_train_class.columns
    )
    scaler_test_class = pd.DataFrame(
        scaler.transform(x_test_class),
        columns = x_test_class.columns
    )
    scaler_train_reg = pd.DataFrame(
        scaler.fit_transform(x_train_reg),
        columns = x_train_reg.columns
    )
    scaler_test_reg = pd.DataFrame(
        scaler.transform(x_test_reg),
        columns = x_test_reg.columns
    )

    joblib.dump(scaler, "artifacts/preprocessor.pkl")

    train_scaled_class = pd.concat(
        [scaler_train_class.reset_index(drop=True), 
        y_train_class.reset_index(drop=True)],
        axis=1
    )
    test_scaled_class = pd.concat(
        [scaler_test_class.reset_index(drop=True), 
        y_test_class.reset_index(drop=True)],
        axis=1
    )
    train_scaled_reg = pd.concat(
        [scaler_train_reg.reset_index(drop=True), 
        y_train_reg.reset_index(drop=True)],
        axis=1
    )
    test_scaled_reg = pd.concat(
        [scaler_test_reg.reset_index(drop=True), 
        y_test_reg.reset_index(drop=True)],
        axis=1
    )

    return train_scaled_class, test_scaled_class, train_scaled_reg, test_scaled_reg

if __name__ == "__main__":
    preprocess()
