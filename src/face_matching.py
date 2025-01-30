import os
import pickle
import numpy as np
from src.feature_extraction import extract_features

# Check current directory and files
print("📂 Current Directory:", os.getcwd())
print("📜 Files:", os.listdir(os.getcwd()))

pickle_path = "face_database.pkl"  # Change to "known_faces.pickle" if needed
image_path = "p1_3.png"

# Load known faces from pickle file
if os.path.exists(pickle_path):
    with open(pickle_path, "rb") as f:
        known_faces = pickle.load(f)
    print("✅ Known faces loaded successfully!")
else:
    print(f"❌ Error: '{pickle_path}' file not found!")
    known_faces = {}  # Avoid crashing if the file is missing


def match_faces(new_image_path, known_faces, threshold=0.6):
    """
    Compares a new face with known faces and returns the closest match.

    Parameters:
    - new_image_path: Path to the new image.
    - known_faces: Dictionary of stored face feature vectors {name: feature_vector}.
    - threshold: Maximum distance for a match (default: 0.6).

    Returns:
    - Best match name or "Unknown"
    """
    new_features = extract_features(new_image_path)  # Extract features from new image

    # Check if the new features are None (no face detected)
    if new_features is None:
        print(f"❌ No face detected in {new_image_path}.")
        return "Unknown", None, None  # Early return if no features were detected

    # Debugging: Print the type of new_features
    print(f"Features extracted for {new_image_path}: {type(new_features)}")

    best_match = None
    min_distance = float("inf")  # Initialize the minimum distance as infinity

    if not known_faces:
        print("❌ No known faces to match against.")
        return "Unknown", None, None

    for name, stored_features in known_faces.items():
        # Ensure stored features are valid
        if stored_features is None:
            print(f"❌ Skipping {name} because its stored features are invalid.")
            continue  # Skip invalid stored features

        # Debugging: Print the type of stored features
        # print(f"Matching {name}: {type(stored_features)}")

        distance = np.linalg.norm(new_features - stored_features)  # Euclidean distance
        similarity = 1 / (1 + distance)  # Euclidean similarity

        if distance < min_distance:  # Update the closest match
            min_distance = distance
            best_match = name
            print(f"Matching {new_image_path} Face with {name}... (Distance: {distance:.4f}, Similarity: {similarity:.4f})")

    # Check if the best match is within the threshold
    if min_distance < threshold:
        print(
            f"✅ Best match to {new_image_path} is: {best_match} (Distance: {min_distance:.4f}, Similarity: {1 / (1 + min_distance):.4f})")
        return best_match, min_distance, 1 / (1 + min_distance)
    else:
        print(f"❌ No match found for {new_image_path} within the threshold.")
        return "Unknown", min_distance, 1 / (1 + min_distance)
