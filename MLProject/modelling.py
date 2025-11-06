import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from dotenv import load_dotenv
# Load data
X_train = pd.read_csv("breastcancer_preprocessing/X_train.csv")
y_train = pd.read_csv("breastcancer_preprocessing/y_train.csv").squeeze()  # .squeeze() agar menjadi Series
X_test = pd.read_csv("breastcancer_preprocessing/X_test.csv")
y_test = pd.read_csv("breastcancer_preprocessing/y_test.csv").squeeze()

# Optional: load scaler/imputer jika ingin pakai preprocessing yang sama
# imputer = joblib.load("imputer.pkl")
# scaler = joblib.load("scaler.pkl")
# X_train = scaler.transform(imputer.transform(X_train))
# X_test = scaler.transform(imputer.transform(X_test))
load_dotenv()

# os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
# os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# mlflow.set_tracking_uri("https://dagshub.com/richellevaniatn/breastcancer-mlflow.mlflow")
# mlflow.set_experiment("BreastCancer_Experiment_Advanced")


with mlflow.start_run(run_name="manual_run"):
    mlflow.sklearn.autolog()

    model = RandomForestRegressor()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
