import os
import numpy as np
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from constants import project_name
import argparse
import pandas as pd
import wandb

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument("--IdExecution", type=str, help="ID of the execution")
args = parser.parse_args()

# Check if the directory "./model" exists
if not os.path.exists("data/input_model"):
    # If it doesn't exist, create it
    os.makedirs("data/input_model")

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")


def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img.mode != "RGBA":
            continue
        try:
            img = img.resize((400, 600)).convert("L")
            img_np = np.array(img)
            images.append(img_np.flatten())  # Flatten the image data
            if "female" in folder:
                labels.append(1)
            else:
                labels.append(0)
        except:
            print(f"Error loading image: {filename}")
        finally:
            img.close()
    return images, labels


def create_images_and_labels(path_folder):
    images, labels = load_images_from_folder(path_folder)
    return images, labels


def create_train_test_data(path_male_faces, path_female_faces):
    male_images, male_labels = create_images_and_labels(path_male_faces)
    female_images, female_labels = create_images_and_labels(path_female_faces)
    X_train_males, X_test_males, y_train_males, y_test_males = train_test_split(
        male_images, male_labels, test_size=0.2, random_state=42
    )
    X_train_females, X_test_females, y_train_females, y_test_females = train_test_split(
        female_images, female_labels, test_size=0.2, random_state=42
    )

    # Combine male and female data
    X_train = X_train_males + X_train_females
    X_test = X_test_males + X_test_females
    y_train = y_train_males + y_train_females
    y_test = y_test_males + y_test_females

    return X_train, y_train, X_test, y_test


def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(
        project=project_name,
        name=f"Load Preproccesed Data ExecId-{args.IdExecution}",
        job_type="load-data",
    ) as run:

        # Load datasets
        X_train, y_train, X_test, y_test = create_train_test_data(
            "data/raw/male_faces/", "data/raw/female_faces/"
        )

        # Convert to pandas DataFrames for easier CSV handling
        train_df = pd.DataFrame(X_train)
        train_df["label"] = y_train

        test_df = pd.DataFrame(X_test)
        test_df["label"] = y_test

        # Save datasets to CSV
        train_df.to_csv("data/input_model/train_data.csv", index=False)
        test_df.to_csv("data/input_model/test_data.csv", index=False)

        # 🏺 Create our Artifacts for W&B
        train_data_artifact = wandb.Artifact(
            "train_data",
            type="dataset",
            description="Training data for gender classification",
            metadata={"source": "custom dataset", "num_samples": len(train_df)},
        )

        test_data_artifact = wandb.Artifact(
            "test_data",
            type="dataset",
            description="Test data for gender classification.",
            metadata={"source": "custom dataset", "num_samples": len(test_df)},
        )

        # Add CSV files to the artifacts
        train_data_artifact.add_file("data/input_model/train_data.csv")
        test_data_artifact.add_file("data/input_model/test_data.csv")

        # ✍️ Log the artifacts to W&B
        run.log_artifact(train_data_artifact)
        run.log_artifact(test_data_artifact)


load_and_log()
