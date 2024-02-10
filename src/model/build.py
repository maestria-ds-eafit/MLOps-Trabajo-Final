from src.knn import knn_model, knn_config

import os
import argparse
import wandb
from constants import project_name
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--IdExecution", type=str, help="ID of the execution", required=True
)
args = parser.parse_args()

print(f"IdExecution: {args.IdExecution}")

# Check if the directory "./model" exists
if not os.path.exists("./model"):
    # If it doesn't exist, create it
    os.makedirs("./model")


def build_model_and_log(config, model, model_name, model_description):
    with wandb.init(
        project=project_name,
        name=f"Initialize KNN Model Execution Id-{args.IdExecution}",
        job_type="initialize-model",
        config=config,
    ) as run:
        config = wandb.config
        model_artifact = wandb.Artifact(
            model_name,
            type="model",
            description=model_description,
            metadata=dict(config),
        )
        name_artifact_model = f"initialized_model_{model_name}.pth"
        pickle.dumps(model, f"./model/{name_artifact_model}")
        model_artifact.add_file(f"./model/{name_artifact_model}")
        wandb.save(name_artifact_model)
        run.log_artifact(model_artifact)


build_model_and_log(knn_config, knn_model, "KNN", "Simple KNN Classifier")
