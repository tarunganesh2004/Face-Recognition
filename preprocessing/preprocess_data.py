import pandas as pd
import os


# Load the dataset
def load_data():
    data = pd.read_csv("data/csv/people.csv")  # Adjust the path if necessary

    # Ensure the 'images' column is an integer (ignore NaN values)
    data["images"] = (
        pd.to_numeric(data["images"], errors="coerce").fillna(0).astype(int)
    )

    # Ensure the 'name' column is treated as a string, replace NaN values with empty strings
    data["name"] = data["name"].astype(str).fillna("")

    image_paths = []
    labels = []

    for index, row in data.iterrows():
        person_name = row["name"]

        # Skip rows where name is empty or invalid
        if person_name == "" or pd.isna(person_name):
            print(
                f"Warning: Skipping row with missing or invalid name at index {index}"
            )
            continue

        num_images = row["images"]

        # Assuming images are stored under 'lfw-deepfunneled' directory
        folder_path = f"data/lfw-deepfunneled/{person_name.replace(' ', '_')}"

        for i in range(1, num_images + 1):
            image_file = f"{person_name.replace(' ', '_')}_{str(i).zfill(4)}.jpg"
            image_path = os.path.join(folder_path, image_file)

            if os.path.exists(image_path):
                image_paths.append(image_path)
                labels.append(person_name)
            else:
                print(f"Warning: {image_path} not found")

    return image_paths, labels


def prepare_data():
    image_paths, labels = load_data()

    # Now you can split your data into train/test or continue with preprocessing
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# Testing the preprocessing function
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
