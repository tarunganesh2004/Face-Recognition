import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def load_lfw_images(data_dir, csv_dir):
    # Load the pairs of images from CSV
    pairs_df = pd.read_csv(os.path.join(csv_dir, "pairs.csv"))

    # Create lists for image pairs and labels
    image_pairs = []
    labels = []

    for index, row in tqdm(pairs_df.iterrows(), total=pairs_df.shape[0]):
        img1 = cv2.imread(os.path.join(data_dir, row["img1"]))
        img2 = cv2.imread(os.path.join(data_dir, row["img2"]))

        # Resize images to 128x128 and normalize
        img1 = cv2.resize(img1, (128, 128)) / 255.0
        img2 = cv2.resize(img2, (128, 128)) / 255.0

        # Add to list
        image_pairs.append([img1, img2])
        labels.append(row["label"])

    return np.array(image_pairs), np.array(labels)


def split_data(image_pairs, labels, test_size=0.2):
    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        image_pairs, labels, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test
