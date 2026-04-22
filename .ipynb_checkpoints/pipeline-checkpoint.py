from data_ingestion import ingest_data
from preprocess import preprocess
from train import train
from evaluation import evaluate

ACCURACY_THRESHOLD = 0.75
F1_THRESHOLD = 0.75
MAE_THRESHOLD = 3
RMSE_THRESHOLD = 4
R2_THRESHOLD = 0.6

def run_pipeline():
    print("Step 1: Data ingestion")
    ingest_data()
    print("Step 2: Preprocessing")

    train_scaled_class, test_scaled_class, train_scaled_reg, test_scaled_reg = preprocess()

    print("Step 3: Training")

    run_id = train(train_scaled_class, train_scaled_reg)

    print("Step 4: Evaluation")

    accuracy, precision, recall, f1, mae, mse, rmse, r2 = evaluate(test_scaled_class, test_scaled_reg, run_id)

    if ((accuracy >= ACCURACY_THRESHOLD) and (f1 >= F1_THRESHOLD) and (mae <= MAE_THRESHOLD) and (rmse <= RMSE_THRESHOLD) and (r2 >= R2_THRESHOLD)):
        print("Model approved for deployment")
    else:
        print("Model rejected")

if __name__ == "__main__":
    run_pipeline()