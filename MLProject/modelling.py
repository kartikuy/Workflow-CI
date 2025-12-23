import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt

# Deteksi apakah berjalan di GitHub Actions
IS_GITHUB = os.getenv("GITHUB_ACTIONS") == "true"

if IS_GITHUB:
    MLFLOW_DB_PATH = "mlflow.db"
    ARTIFACT_ROOT = "mlruns"
    BASE_DIR = "."
else:
    # Path normal versi Windows/lokal
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MLFLOW_DB_PATH = os.path.join(BASE_DIR, "mlflow.db")
    ARTIFACT_ROOT = os.path.join(BASE_DIR, "mlruns")

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
mlflow.set_experiment("Bank Customer Churn Experiment")

dataset_path = os.path.join(BASE_DIR,  "preprocessed_Bank_Customer_Churn_Prediction.csv")

df = pd.read_csv(dataset_path)

y = df["churn"]
X = df.drop(columns=["churn"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mlflow.sklearn.autolog()

rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf.fit(X_train, y_train)

accuracy = rf.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")