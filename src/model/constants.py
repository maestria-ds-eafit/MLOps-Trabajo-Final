from sklearn.metrics import accuracy_score
import pandas as pd

experiment_id = 1
project_name = "MLOps-Trabajo-Final"


def read_csv(data_dir, filename):
    return pd.read_csv(f"{data_dir}/{filename}")


def log_metrics(wandbRun, y_true, y_pred, dryRun=True):
    accuracy = accuracy_score(y_true, y_pred)
    if not dryRun:
        wandbRun.summary["accuracy"] = accuracy
