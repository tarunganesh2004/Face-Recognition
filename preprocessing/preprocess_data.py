import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

# Paths
DATA_PATH = "./data/lfw-deepfunneled/"
CSV_PATH = "./data/csv/people.csv"


# Load CSV file with names and labels
def load_data():
    people_df = pd.read_csv(CSV_PATH)

    image_paths = []
    labels = []

    for _, row in people_df.iterrows():
        person_name = row["Name"]  # Assuming CSV has a 'Name' column
        label = row["Label"]  # Assuming there's a 'Label' column

        # For each person, get all images
        person_dir = os.path.join(DATA_PATH, person_name)
        if os.path.exists(person_dir):
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                image_paths.append(img_path)
                labels.append(label)

    return image_paths, labels


# Preprocess the images (resize and normalize)
def preprocess_image(image_path, img_size=(128, 128)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)
    img = img.astype("float32") / 255.0  # Normalize
    return img


# Prepare data for training
def prepare_data(img_size=(128, 128)):
    image_paths, labels = load_data()

    X = []
    y = []

    for img_path, label in zip(image_paths, labels):
        img = preprocess_image(img_path, img_size)
        X.append(img)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"Training samples: {X_train.shape}, Testing samples: {X_test.shape}")
