import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,ConfusionMatrixDisplay
from constants import project_name
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--IdExecution", type=str, help="ID of the execution")
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")


def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).resize((400, 600)).convert("L")
        img = np.array(img)
        images.append(img.flatten())  # Flatten the image data
        if "female" in folder:
            labels.append(1)
        else:
            labels.append(0)
    return images, labels

def create_images_and_labels(path_folder):
    images, labels = load_images_from_folder(path_folder)
    return images, labels

def create_train_test_data(path_male_faces, path_female_faces):
    male_images, male_labels = create_images_and_labels('../data/raw/male Faces/')
    female_images, female_labels = create_images_and_labels('../data/raw/female Faces/')
    X_train_males, X_test_males, y_train_males, y_test_males = train_test_split(male_images, male_labels, test_size=0.2, random_state=42)
    X_train_females, X_test_females, y_train_females, y_test_females = train_test_split(female_images, female_labels, test_size=0.2, random_state=42)

    # Combine male and female data
    X_train=X_train_males+X_train_females
    X_test=X_test_males+X_test_females
    y_train=y_train_males+y_train_females
    y_test=y_test_males+y_test_females

    return X_train, y_train, X_test, y_test

def load(train_size=0.8):
    """
    # Load the data
    """

    # the data, split between train and test sets
    train = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    test = torchvision.datasets.MNIST(root="./data", train=False, download=True)

    (x_train, y_train), (x_test, y_test) = (train.data, train.targets), (
        test.data,
        test.targets,
    )

    # split off a validation set for hyperparameter tuning
    x_train, x_val = (
        x_train[: int(len(train) * train_size)],
        x_train[int(len(train) * train_size) :],
    )
    y_train, y_val = (
        y_train[: int(len(train) * train_size)],
        y_train[int(len(train) * train_size) :],
    )

    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)
    datasets = [training_set, validation_set, test_set]
    return datasets

import pandas as pd
import wandb

# Assuming the rest of your code is as provided above

def load_and_log():
    # 🚀 Start a W&B run
    with wandb.init(
        project=project_name,
        name=f"Load Data ExecId-{args.IdExecution}", job_type="load-data") as run:
        # Load datasets
        X_train, y_train, X_test, y_test = create_train_test_data('../data/raw/male_faces', '../data/raw/female_faces')
        
        # Convert to pandas DataFrames for easier CSV handling
        train_df = pd.DataFrame(X_train)
        train_df['label'] = y_train
        
        test_df = pd.DataFrame(X_test)
        test_df['label'] = y_test
        
        # Save datasets to CSV
        train_df.to_csv('train_data.csv', index=False)
        test_df.to_csv('test_data.csv', index=False)
        
        # 🏺 Create our Artifacts for W&B
        train_data_artifact = wandb.Artifact("train_data", type="dataset", description="Training data for gender classification", metadata={"source": "custom dataset", "num_samples": len(train_df)})
        test_data_artifact = wandb.Artifact("test_data", type="dataset", description="Test data for gender classification", metadata={"source": "custom dataset", "num_samples": len(test_df)})

        # Add CSV files to the artifacts
        train_data_artifact.add_file('train_data.csv')
        test_data_artifact.add_file('test_data.csv')
        
        # ✍️ Log the artifacts to W&B
        run.log_artifact(train_data_artifact)
        run.log_artifact(test_data_artifact)

# Call the function to execute
load_and_log()


def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(
        project=project_name,
        name=f"Load Preproccesed Data ExecId-{args.IdExecution}",
        job_type="load-data",
    ) as run:

        # Load datasets
        X_train, y_train, X_test, y_test = create_train_test_data('../data/raw/male_faces', '../data/raw/female_faces')
        
        # Convert to pandas DataFrames for easier CSV handling
        train_df = pd.DataFrame(X_train)
        train_df['label'] = y_train
        
        test_df = pd.DataFrame(X_test)
        test_df['label'] = y_test
        
        # Save datasets to CSV
        train_df.to_csv('train_data.csv', index=False)
        test_df.to_csv('test_data.csv', index=False)
        
        # 🏺 Create our Artifacts for W&B
        train_data_artifact = wandb.Artifact("train_data", type="dataset", description="Training data for gender classification", metadata={"source": "custom dataset", "num_samples": len(train_df)})
        test_data_artifact = wandb.Artifact("test_data", type="dataset", description="Test data for gender classification", metadata={"source": "custom dataset", "num_samples": len(test_df)})

        # Add CSV files to the artifacts
        train_data_artifact.add_file('train_data.csv')
        test_data_artifact.add_file('test_data.csv')
        
        # ✍️ Log the artifacts to W&B
        run.log_artifact(train_data_artifact)
        run.log_artifact(test_data_artifact)

        run.log_artifact(raw_data)




