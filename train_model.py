from keras_facenet import FaceNet
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from preprocessing.preprocess_data import prepare_data

# Initialize FaceNet
embedder = FaceNet()


def get_image_embedding(img_path):
    # Load and preprocess the image
    img = image.load_img(
        img_path, target_size=(160, 160)
    )  # FaceNet requires 160x160 images
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Get the embedding
    embedding = embedder.embeddings(img_array)
    return embedding.flatten()


def extract_embeddings(image_paths):
    embeddings = []
    for path in image_paths:
        embedding = get_image_embedding(path)
        embeddings.append(embedding)
    return np.array(embeddings)


def train_model():
    X_train, X_test, y_train, y_test = prepare_data()

    # Extract embeddings for both train and test datasets
    print("Extracting embeddings for training data...")
    embeddings_train = extract_embeddings(X_train)
    print("Extracting embeddings for test data...")
    embeddings_test = extract_embeddings(X_test)

    # Encode the labels (person names)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Train an SVM classifier
    print("Training SVM classifier...")
    svm_model = SVC(kernel="linear", probability=True)
    svm_model.fit(embeddings_train, y_train_encoded)

    # Save the trained model and label encoder
    joblib.dump(svm_model, "models/face_recognition_svm_model.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    # Evaluate the model on the test set
    y_pred = svm_model.predict(embeddings_test)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    train_model()
